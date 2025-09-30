#!/usr/bin/env python3
"""
Stoma Batch Processing System - Overnight Analysis Workflows

This module provides comprehensive batch processing capabilities for large-scale
analysis workflows. It builds on the minimal pipeline but adds scheduling,
batching, and advanced features for production deployment.
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
import hashlib
import sqlite3

# Import our existing components
from minimal_pipeline import (
    MinimalArXivCollector,
    MinimalLLMAnalyzer,
    MinimalHealthChecker,
    MinimalReportStorage
)
from advanced_model_manager import AdvancedModelManager

logger = logging.getLogger(__name__)


@dataclass
class BatchJob:
    """Represents a batch processing job."""
    job_id: str
    job_type: str  # 'arxiv_analysis', 'cross_paper_synthesis', etc.
    parameters: Dict[str, Any]
    priority: int = 1  # 1=high, 5=low
    scheduled_time: Optional[datetime] = None
    started_time: Optional[datetime] = None
    completed_time: Optional[datetime] = None
    status: str = "pending"  # pending, running, completed, failed
    results_path: Optional[str] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class BatchProcessingConfig:
    """Configuration for batch processing operations."""
    max_concurrent_jobs: int = 3
    max_papers_per_job: int = 50
    default_model: str = "gemma2:2b"
    advanced_model: str = "llama3.1:8b"
    heavy_model: str = "llama3.1:70b"
    storage_path: str = "./reports/batch_processing"
    job_timeout_minutes: int = 180  # 3 hours per job
    enable_auto_scheduling: bool = True
    overnight_start_hour: int = 22  # 10 PM
    overnight_end_hour: int = 6    # 6 AM


class BatchJobScheduler:
    """Manages scheduling and queuing of batch jobs."""

    def __init__(self, config: BatchProcessingConfig):
        self.config = config
        self.job_queue: List[BatchJob] = []
        self.running_jobs: Dict[str, BatchJob] = {}
        self.completed_jobs: List[BatchJob] = []
        self.storage_path = Path(config.storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.jobs_db_path = self.storage_path / "batch_jobs.db"

        # Initialize job tracking database
        self._init_jobs_database()

        # Load existing jobs from database
        self._load_jobs_from_database()

    def _init_jobs_database(self):
        """Initialize SQLite database for job tracking."""
        try:
            conn = sqlite3.connect(self.jobs_db_path)
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS batch_jobs (
                    job_id TEXT PRIMARY KEY,
                    job_type TEXT NOT NULL,
                    parameters TEXT NOT NULL,
                    priority INTEGER DEFAULT 1,
                    scheduled_time TEXT,
                    started_time TEXT,
                    completed_time TEXT,
                    status TEXT DEFAULT 'pending',
                    results_path TEXT,
                    error_message TEXT,
                    retry_count INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.commit()
            conn.close()
            logger.info("Batch jobs database initialized")

        except Exception as e:
            logger.error(f"Failed to initialize jobs database: {e}")

    def _load_jobs_from_database(self):
        """Load existing jobs from database."""
        try:
            conn = sqlite3.connect(self.jobs_db_path)
            cursor = conn.cursor()

            cursor.execute("SELECT * FROM batch_jobs WHERE status IN ('pending', 'running')")
            rows = cursor.fetchall()

            for row in rows:
                job = BatchJob(
                    job_id=row[0],
                    job_type=row[1],
                    parameters=json.loads(row[2]),
                    priority=row[3],
                    scheduled_time=datetime.fromisoformat(row[4]) if row[4] else None,
                    started_time=datetime.fromisoformat(row[5]) if row[5] else None,
                    completed_time=datetime.fromisoformat(row[6]) if row[6] else None,
                    status=row[7],
                    results_path=row[8],
                    error_message=row[9],
                    retry_count=row[10]
                )

                if job.status == 'pending':
                    self.job_queue.append(job)
                elif job.status == 'running':
                    # Mark running jobs as pending if we're restarting
                    job.status = 'pending'
                    self.job_queue.append(job)

            conn.close()
            logger.info(f"Loaded {len(self.job_queue)} jobs from database")

        except Exception as e:
            logger.error(f"Failed to load jobs from database: {e}")

    def schedule_job(self, job: BatchJob) -> str:
        """Schedule a new batch job."""
        job.job_id = self._generate_job_id(job.job_type)

        # Save to database
        self._save_job_to_database(job)

        # Add to queue
        self.job_queue.append(job)
        self.job_queue.sort(key=lambda x: (x.priority, x.scheduled_time or datetime.min))

        logger.info(f"Scheduled job {job.job_id} ({job.job_type}) with priority {job.priority}")
        return job.job_id

    def schedule_arxiv_analysis(self, query: str, max_papers: int = 20,
                               model: str = None, priority: int = 1) -> str:
        """Schedule an ArXiv analysis job."""
        job = BatchJob(
            job_id="",  # Will be set by schedule_job
            job_type="arxiv_analysis",
            parameters={
                "query": query,
                "max_papers": max_papers,
                "model": model or self.config.default_model
            },
            priority=priority
        )
        return self.schedule_job(job)

    def schedule_cross_paper_synthesis(self, query_set: List[str],
                                     synthesis_type: str = "trending_themes",
                                     priority: int = 2) -> str:
        """Schedule a cross-paper synthesis job."""
        job = BatchJob(
            job_id="",
            job_type="cross_paper_synthesis",
            parameters={
                "query_set": query_set,
                "synthesis_type": synthesis_type,
                "model": self.config.advanced_model
            },
            priority=priority
        )
        return self.schedule_job(job)

    def schedule_overnight_analysis_batch(self) -> List[str]:
        """Schedule a comprehensive overnight analysis batch."""
        # Define research areas for comprehensive analysis
        research_queries = [
            "large language models",
            "computer vision transformers",
            "reinforcement learning agents",
            "quantum computing algorithms",
            "neural architecture search",
            "federated learning",
            "graph neural networks",
            "protein folding prediction",
            "autonomous vehicle perception",
            "medical imaging AI"
        ]

        job_ids = []

        # Schedule individual analysis jobs
        for i, query in enumerate(research_queries):
            job_id = self.schedule_arxiv_analysis(
                query=query,
                max_papers=15,
                model=self.config.default_model,
                priority=3
            )
            job_ids.append(job_id)

        # Schedule synthesis job to run after individual analyses
        synthesis_job_id = self.schedule_cross_paper_synthesis(
            query_set=research_queries,
            synthesis_type="cross_domain_trends",
            priority=4  # Run after individual analyses
        )
        job_ids.append(synthesis_job_id)

        logger.info(f"Scheduled overnight batch: {len(job_ids)} jobs")
        return job_ids

    def get_next_job(self) -> Optional[BatchJob]:
        """Get the next job to execute."""
        if not self.job_queue:
            return None

        # Check if we have room for more concurrent jobs
        if len(self.running_jobs) >= self.config.max_concurrent_jobs:
            return None

        # Find highest priority job that's ready to run
        now = datetime.now()
        for i, job in enumerate(self.job_queue):
            if job.scheduled_time is None or job.scheduled_time <= now:
                return self.job_queue.pop(i)

        return None

    def start_job(self, job: BatchJob):
        """Mark a job as started."""
        job.status = "running"
        job.started_time = datetime.now()
        self.running_jobs[job.job_id] = job
        self._save_job_to_database(job)
        logger.info(f"Started job {job.job_id}")

    def complete_job(self, job: BatchJob, results_path: str = None, error: str = None):
        """Mark a job as completed."""
        job.completed_time = datetime.now()
        job.results_path = results_path
        job.error_message = error
        job.status = "completed" if not error else "failed"

        if job.job_id in self.running_jobs:
            del self.running_jobs[job.job_id]

        self.completed_jobs.append(job)
        self._save_job_to_database(job)

        status_msg = f"Completed job {job.job_id}"
        if error:
            status_msg += f" with error: {error}"
        logger.info(status_msg)

    def get_job_status(self) -> Dict[str, Any]:
        """Get current status of all jobs."""
        return {
            "queued_jobs": len(self.job_queue),
            "running_jobs": len(self.running_jobs),
            "completed_jobs": len(self.completed_jobs),
            "queue_summary": [
                {"job_id": job.job_id, "type": job.job_type, "priority": job.priority}
                for job in self.job_queue[:10]  # Show first 10
            ],
            "running_summary": [
                {
                    "job_id": job.job_id,
                    "type": job.job_type,
                    "started": job.started_time.isoformat() if job.started_time else None,
                    "duration_minutes": (datetime.now() - job.started_time).total_seconds() / 60 if job.started_time else 0
                }
                for job in self.running_jobs.values()
            ]
        }

    def _generate_job_id(self, job_type: str) -> str:
        """Generate a unique job ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{job_type}_{timestamp}_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}"

    def _save_job_to_database(self, job: BatchJob):
        """Save job to database."""
        try:
            conn = sqlite3.connect(self.jobs_db_path)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT OR REPLACE INTO batch_jobs
                (job_id, job_type, parameters, priority, scheduled_time, started_time,
                 completed_time, status, results_path, error_message, retry_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                job.job_id,
                job.job_type,
                json.dumps(job.parameters),
                job.priority,
                job.scheduled_time.isoformat() if job.scheduled_time else None,
                job.started_time.isoformat() if job.started_time else None,
                job.completed_time.isoformat() if job.completed_time else None,
                job.status,
                job.results_path,
                job.error_message,
                job.retry_count
            ))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to save job to database: {e}")


class BatchProcessor:
    """Main batch processor that executes scheduled jobs."""

    def __init__(self, config: BatchProcessingConfig):
        self.config = config
        self.scheduler = BatchJobScheduler(config)
        self.health_checker = MinimalHealthChecker(auto_repair=True)
        self.storage = MinimalReportStorage()
        self.model_manager = AdvancedModelManager()
        self.running = False

    async def run_batch_processor(self, max_iterations: int = None):
        """Run the batch processor main loop."""
        logger.info("Starting batch processor")
        self.running = True
        iterations = 0

        # Perform initial health check
        logger.info("Running initial system health check...")
        health_status = await self.health_checker.check_all_systems()

        if not health_status['arxiv']:
            logger.error("ArXiv API is not available - stopping batch processor")
            return

        if not health_status['ollama'] or not health_status['models']:
            logger.error("Ollama or models not available - stopping batch processor")
            return

        logger.info("System health check passed - starting job processing")

        while self.running:
            try:
                # Get next job
                job = self.scheduler.get_next_job()

                if job:
                    await self._execute_job(job)
                else:
                    # No jobs available, wait a bit
                    await asyncio.sleep(30)

                iterations += 1
                if max_iterations and iterations >= max_iterations:
                    logger.info(f"Reached max iterations ({max_iterations})")
                    break

                # Log status every 10 minutes
                if iterations % 20 == 0:  # 30s * 20 = 10 minutes
                    status = self.scheduler.get_job_status()
                    logger.info(f"Status: {status['queued_jobs']} queued, {status['running_jobs']} running, {status['completed_jobs']} completed")

            except Exception as e:
                logger.error(f"Error in batch processor main loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying

        logger.info("Batch processor stopped")

    async def _execute_job(self, job: BatchJob):
        """Execute a specific job."""
        logger.info(f"Executing job {job.job_id} ({job.job_type})")

        self.scheduler.start_job(job)

        try:
            if job.job_type == "arxiv_analysis":
                await self._execute_arxiv_analysis(job)
            elif job.job_type == "cross_paper_synthesis":
                await self._execute_cross_paper_synthesis(job)
            else:
                raise ValueError(f"Unknown job type: {job.job_type}")

        except Exception as e:
            logger.error(f"Job {job.job_id} failed: {e}")
            self.scheduler.complete_job(job, error=str(e))

    async def _execute_arxiv_analysis(self, job: BatchJob):
        """Execute an ArXiv analysis job."""
        params = job.parameters
        query = params["query"]
        max_papers = params["max_papers"]
        model = params["model"]

        logger.info(f"Analyzing ArXiv papers: query='{query}', max_papers={max_papers}, model={model}")

        # Collect papers
        collector = MinimalArXivCollector(max_papers)
        papers = await collector.collect(query)

        if not papers:
            raise Exception("No papers collected")

        # Prepare content for analysis
        analysis_content = []
        for i, paper in enumerate(papers[:min(10, len(papers))], 1):
            content = f"Paper {i}:\nTitle: {paper['title']}\nAbstract: {paper['abstract']}\n"
            analysis_content.append(content)

        combined_text = "\n".join(analysis_content)

        # Select optimal model for analysis task
        optimal_model = self.model_manager.select_model_for_task(
            task_type="research_analysis",
            content_length=len(combined_text)
        )
        logger.info(f"Selected model {optimal_model} for analysis (requested: {model})")

        # Run LLM analysis with optimal model
        analyzer = MinimalLLMAnalyzer(optimal_model)
        analysis = await analyzer.analyze(combined_text, f"Batch Analysis: {query}")

        if not analysis:
            raise Exception("LLM analysis failed")

        # Save results
        timestamp = datetime.now()
        filename = f"batch_{job.job_id}_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"

        report = {
            "job_id": job.job_id,
            "job_type": job.job_type,
            "timestamp": timestamp.isoformat(),
            "query": query,
            "model": model,
            "papers_count": len(papers),
            "analysis": analysis,
            "papers": papers,
            "processing_time_minutes": (datetime.now() - job.started_time).total_seconds() / 60
        }

        results_path = self.storage.storage_path / filename
        with open(results_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Batch analysis complete: {len(papers)} papers, saved to {filename}")
        self.scheduler.complete_job(job, results_path=str(results_path))

    async def _execute_cross_paper_synthesis(self, job: BatchJob):
        """Execute a cross-paper synthesis job."""
        params = job.parameters
        query_set = params["query_set"]
        synthesis_type = params["synthesis_type"]
        model = params["model"]

        logger.info(f"Running cross-paper synthesis: {synthesis_type}, queries={len(query_set)}")

        # Import the synthesizer
        from cross_paper_synthesizer import CrossPaperSynthesizer, ResearchPaper

        # Collect papers for each query
        all_papers = []
        collector = MinimalArXivCollector(max_results=10)  # 10 papers per query

        for query in query_set:
            try:
                papers = await collector.collect(query)
                for i, paper_data in enumerate(papers):
                    paper = ResearchPaper(
                        paper_id=f"{query}_{i}",
                        title=paper_data['title'],
                        abstract=paper_data.get('abstract', ''),
                        authors=paper_data.get('authors', []),
                        published_date=paper_data.get('published', ''),
                        arxiv_id=paper_data.get('arxiv_id', ''),
                        url=paper_data.get('url', ''),
                        domain=query.replace(' ', '_'),  # Simple domain mapping
                        keywords=[],  # Will be filled by classifier
                        analysis_result=None  # Will be filled by LLM analysis if needed
                    )
                    all_papers.append(paper)

            except Exception as e:
                logger.warning(f"Failed to collect papers for query '{query}': {e}")

        if not all_papers:
            raise Exception("No papers collected for synthesis")

        # Select optimal model for synthesis task
        optimal_model = self.model_manager.select_model_for_task(
            task_type="synthesis",
            content_length=sum(len(p.abstract) for p in all_papers),
            performance_preference="quality"
        )
        logger.info(f"Selected model {optimal_model} for synthesis (requested: {model})")

        # Run synthesis with optimal model
        synthesizer = CrossPaperSynthesizer(model=optimal_model)
        synthesis_result_obj = await synthesizer.synthesize_papers(all_papers)

        # Convert to serializable format
        synthesis_result = {
            "job_id": job.job_id,
            "synthesis_type": synthesis_type,
            "query_set": query_set,
            "model": model,
            "timestamp": synthesis_result_obj.synthesis_timestamp.isoformat(),
            "papers_analyzed": synthesis_result_obj.papers_analyzed,
            "synthesis_quality_score": synthesis_result_obj.synthesis_quality_score,
            "cross_domain_trends": [
                {
                    "trend_name": trend.trend_name,
                    "description": trend.description,
                    "confidence_score": trend.confidence_score,
                    "trend_type": trend.trend_type,
                    "supporting_papers": len(trend.supporting_papers)
                }
                for trend in synthesis_result_obj.cross_domain_trends
            ],
            "common_techniques": synthesis_result_obj.common_techniques,
            "emerging_themes": synthesis_result_obj.emerging_themes,
            "research_gaps": synthesis_result_obj.research_gaps,
            "recommendations": synthesis_result_obj.recommendations,
            "limitations": synthesis_result_obj.limitations
        }

        # Save synthesis results
        timestamp = datetime.now()
        filename = f"synthesis_{job.job_id}_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        results_path = self.storage.storage_path / filename

        with open(results_path, 'w') as f:
            json.dump(synthesis_result, f, indent=2, default=str)

        logger.info(f"Cross-paper synthesis complete, saved to {filename}")
        self.scheduler.complete_job(job, results_path=str(results_path))

    def stop_processor(self):
        """Stop the batch processor."""
        logger.info("Stopping batch processor...")
        self.running = False


# CLI Interface for Batch Processing

def create_default_config() -> BatchProcessingConfig:
    """Create default batch processing configuration."""
    return BatchProcessingConfig(
        max_concurrent_jobs=int(os.getenv("BATCH_MAX_CONCURRENT", "3")),
        default_model=os.getenv("BATCH_DEFAULT_MODEL", "gemma2:2b"),
        advanced_model=os.getenv("BATCH_ADVANCED_MODEL", "llama3.1:8b"),
        storage_path=os.getenv("BATCH_STORAGE_PATH", "./reports/batch_processing")
    )


async def main():
    """Main CLI entry point."""
    if len(sys.argv) < 2:
        print("Usage: python3 batch_processor.py <command> [options]")
        print("Commands:")
        print("  run                     - Start batch processor daemon")
        print("  schedule <query>        - Schedule ArXiv analysis")
        print("  overnight               - Schedule overnight analysis batch")
        print("  status                  - Show job status")
        print("  health                  - Run health check")
        print("Examples:")
        print("  python3 batch_processor.py run")
        print("  python3 batch_processor.py schedule 'machine learning'")
        print("  python3 batch_processor.py overnight")
        return

    config = create_default_config()
    processor = BatchProcessor(config)

    command = sys.argv[1].lower()

    if command == "run":
        # Run the batch processor daemon
        try:
            await processor.run_batch_processor()
        except KeyboardInterrupt:
            processor.stop_processor()
            print("\nBatch processor stopped")

    elif command == "schedule":
        if len(sys.argv) < 3:
            print("Usage: python3 batch_processor.py schedule <query> [max_papers]")
            return

        query = sys.argv[2]
        max_papers = int(sys.argv[3]) if len(sys.argv) > 3 else 20

        job_id = processor.scheduler.schedule_arxiv_analysis(query, max_papers)
        print(f"Scheduled ArXiv analysis job: {job_id}")
        print(f"Query: {query}, Max papers: {max_papers}")

    elif command == "overnight":
        job_ids = processor.scheduler.schedule_overnight_analysis_batch()
        print(f"Scheduled overnight analysis batch: {len(job_ids)} jobs")
        for job_id in job_ids:
            print(f"  - {job_id}")

    elif command == "status":
        status = processor.scheduler.get_job_status()
        print("Batch Processing Status:")
        print(f"  Queued jobs: {status['queued_jobs']}")
        print(f"  Running jobs: {status['running_jobs']}")
        print(f"  Completed jobs: {status['completed_jobs']}")

        if status['queue_summary']:
            print("\nNext jobs in queue:")
            for job in status['queue_summary']:
                print(f"  - {job['job_id']} ({job['type']}, priority {job['priority']})")

        if status['running_summary']:
            print("\nCurrently running:")
            for job in status['running_summary']:
                print(f"  - {job['job_id']} ({job['type']}, running {job['duration_minutes']:.1f} min)")

    elif command == "health":
        health_checker = MinimalHealthChecker(auto_repair=True)
        await health_checker.check_all_systems()

    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    asyncio.run(main())