"""Overnight batch processing system for analysis tasks."""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import time

from .nlp_service import NLPService
from .trend_detector import TrendDetector
from .correlation_analyzer import CorrelationAnalyzer
from ..storage.database import DatabaseStorage

logger = logging.getLogger(__name__)


@dataclass
class BatchTask:
    """Represents a batch processing task."""
    
    task_id: str
    task_type: str  # 'nlp_analysis', 'trend_detection', 'correlation_analysis'
    priority: int  # 1-10, higher is more important
    max_runtime_minutes: int
    parameters: Dict[str, Any]
    created_at: datetime
    scheduled_for: datetime
    status: str = 'pending'  # 'pending', 'running', 'completed', 'failed'
    result: Optional[Dict] = None
    error_message: Optional[str] = None


@dataclass
class BatchJobResult:
    """Results from a batch processing job."""
    
    job_id: str
    start_time: datetime
    end_time: datetime
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    skipped_tasks: int
    total_processing_time: float
    results_summary: Dict[str, Any]


class BatchProcessor:
    """Manages overnight batch processing of analysis tasks."""
    
    def __init__(self, 
                 db_storage: Optional[DatabaseStorage] = None,
                 max_concurrent_tasks: int = 3):
        """
        Initialize batch processor.
        
        Args:
            db_storage: Database storage instance
            max_concurrent_tasks: Maximum tasks to run concurrently
        """
        self.db = db_storage or DatabaseStorage()
        self.max_concurrent_tasks = max_concurrent_tasks
        
        # Initialize services
        self.nlp_service = NLPService(self.db)
        self.trend_detector = TrendDetector(self.db)
        self.correlation_analyzer = CorrelationAnalyzer(self.db)
        
        # Task queue
        self.task_queue: List[BatchTask] = []
        self.running_tasks: Dict[str, BatchTask] = {}
        
        logger.info("Batch processor initialized")
    
    def schedule_nlp_analysis_batch(self, 
                                   paper_ids: Optional[List[int]] = None,
                                   priority: int = 5,
                                   max_papers: int = 100) -> str:
        """
        Schedule batch NLP analysis of papers.
        
        Args:
            paper_ids: Specific papers to analyze, or None for unanalyzed papers
            priority: Task priority (1-10)
            max_papers: Maximum papers to process
            
        Returns:
            Task ID
        """
        task_id = f"nlp_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        task = BatchTask(
            task_id=task_id,
            task_type='nlp_analysis',
            priority=priority,
            max_runtime_minutes=120,  # 2 hours max
            parameters={
                'paper_ids': paper_ids,
                'max_papers': max_papers
            },
            created_at=datetime.now(),
            scheduled_for=self._get_next_batch_window()
        )
        
        self.task_queue.append(task)
        logger.info(f"Scheduled NLP analysis batch: {task_id}")
        
        return task_id
    
    def schedule_trend_analysis_batch(self, 
                                     timeframe_days: int = 30,
                                     priority: int = 6) -> str:
        """
        Schedule batch trend analysis.
        
        Args:
            timeframe_days: Time window for trend analysis
            priority: Task priority
            
        Returns:
            Task ID
        """
        task_id = f"trend_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        task = BatchTask(
            task_id=task_id,
            task_type='trend_detection',
            priority=priority,
            max_runtime_minutes=60,
            parameters={
                'timeframe_days': timeframe_days,
                'include_cross_domain': True,
                'include_emerging_topics': True
            },
            created_at=datetime.now(),
            scheduled_for=self._get_next_batch_window()
        )
        
        self.task_queue.append(task)
        logger.info(f"Scheduled trend analysis batch: {task_id}")
        
        return task_id
    
    def schedule_correlation_analysis_batch(self, 
                                          paper_ids: Optional[List[int]] = None,
                                          priority: int = 4) -> str:
        """
        Schedule batch correlation analysis.
        
        Args:
            paper_ids: Papers to analyze, or None for recent papers
            priority: Task priority
            
        Returns:
            Task ID
        """
        task_id = f"correlation_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        task = BatchTask(
            task_id=task_id,
            task_type='correlation_analysis',
            priority=priority,
            max_runtime_minutes=90,
            parameters={
                'paper_ids': paper_ids,
                'correlation_threshold': 0.3,
                'max_correlations': 200,
                'include_clustering': True
            },
            created_at=datetime.now(),
            scheduled_for=self._get_next_batch_window()
        )
        
        self.task_queue.append(task)
        logger.info(f"Scheduled correlation analysis batch: {task_id}")
        
        return task_id
    
    async def run_batch_job(self, 
                           job_id: Optional[str] = None,
                           max_runtime_hours: int = 6) -> BatchJobResult:
        """
        Run a batch processing job.
        
        Args:
            job_id: Optional job identifier
            max_runtime_hours: Maximum runtime for entire job
            
        Returns:
            Batch job results
        """
        if not job_id:
            job_id = f"batch_job_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        start_time = datetime.now()
        max_end_time = start_time + timedelta(hours=max_runtime_hours)
        
        logger.info(f"Starting batch job: {job_id}")
        
        # Sort tasks by priority and scheduled time
        pending_tasks = [t for t in self.task_queue if t.status == 'pending']
        pending_tasks.sort(key=lambda x: (-x.priority, x.scheduled_for))
        
        total_tasks = len(pending_tasks)
        completed_tasks = 0
        failed_tasks = 0
        skipped_tasks = 0
        
        # Process tasks
        for task in pending_tasks:
            # Check if we've exceeded max runtime
            if datetime.now() >= max_end_time:
                logger.warning(f"Batch job {job_id} exceeded max runtime, stopping")
                skipped_tasks = total_tasks - completed_tasks - failed_tasks
                break
            
            # Check if we can start more tasks
            if len(self.running_tasks) >= self.max_concurrent_tasks:
                # Wait for a task to complete
                await self._wait_for_task_completion()
            
            # Run the task
            try:
                await self._run_task(task)
                completed_tasks += 1
            except Exception as e:
                logger.error(f"Task {task.task_id} failed: {e}")
                task.status = 'failed'
                task.error_message = str(e)
                failed_tasks += 1
        
        # Wait for remaining tasks to complete
        while self.running_tasks:
            await self._wait_for_task_completion()
        
        end_time = datetime.now()
        total_processing_time = (end_time - start_time).total_seconds()
        
        # Compile results
        results_summary = self._compile_batch_results(pending_tasks)
        
        result = BatchJobResult(
            job_id=job_id,
            start_time=start_time,
            end_time=end_time,
            total_tasks=total_tasks,
            completed_tasks=completed_tasks,
            failed_tasks=failed_tasks,
            skipped_tasks=skipped_tasks,
            total_processing_time=total_processing_time,
            results_summary=results_summary
        )
        
        logger.info(
            f"Batch job {job_id} completed: {completed_tasks} success, "
            f"{failed_tasks} failed, {skipped_tasks} skipped in "
            f"{total_processing_time:.1f}s"
        )
        
        return result
    
    async def _run_task(self, task: BatchTask):
        """Run a single batch task."""
        logger.info(f"Starting task: {task.task_id} ({task.task_type})")
        
        task.status = 'running'
        self.running_tasks[task.task_id] = task
        start_time = time.time()
        
        try:
            if task.task_type == 'nlp_analysis':
                result = await self._run_nlp_analysis_task(task)
            elif task.task_type == 'trend_detection':
                result = await self._run_trend_detection_task(task)
            elif task.task_type == 'correlation_analysis':
                result = await self._run_correlation_analysis_task(task)
            else:
                raise ValueError(f"Unknown task type: {task.task_type}")
            
            task.result = result
            task.status = 'completed'
            
            processing_time = time.time() - start_time
            logger.info(f"Task {task.task_id} completed in {processing_time:.1f}s")
            
        except Exception as e:
            task.status = 'failed'
            task.error_message = str(e)
            logger.error(f"Task {task.task_id} failed: {e}")
            raise
        
        finally:
            self.running_tasks.pop(task.task_id, None)
    
    async def _run_nlp_analysis_task(self, task: BatchTask) -> Dict:
        """Run NLP analysis task."""
        params = task.parameters
        
        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        
        result = await loop.run_in_executor(
            None,
            self.nlp_service.batch_analyze_papers,
            params.get('paper_ids'),
            params.get('max_papers', 100)
        )
        
        return {
            'task_type': 'nlp_analysis',
            'papers_processed': result.get('total_processed', 0),
            'successful_analyses': result.get('successful', 0),
            'failed_analyses': result.get('failed', 0),
            'processing_time_ms': result.get('processing_time_ms', 0)
        }
    
    async def _run_trend_detection_task(self, task: BatchTask) -> Dict:
        """Run trend detection task."""
        params = task.parameters
        
        loop = asyncio.get_event_loop()
        
        # Run keyword trends analysis
        keyword_trends = await loop.run_in_executor(
            None,
            self.trend_detector.detect_keyword_trends,
            params.get('timeframe_days', 30),
            3  # min_frequency
        )
        
        # Run emerging topics analysis if requested
        emerging_topics = []
        if params.get('include_emerging_topics', False):
            emerging_topics = await loop.run_in_executor(
                None,
                self.trend_detector.detect_emerging_topics,
                params.get('timeframe_days', 30) * 3,  # longer lookback
                0.5  # emergence_threshold
            )
        
        # Run cross-domain analysis if requested
        cross_domain_trends = []
        if params.get('include_cross_domain', False):
            cross_domain_trends = await loop.run_in_executor(
                None,
                self.trend_detector.detect_cross_domain_trends,
                params.get('timeframe_days', 30) * 2
            )
        
        return {
            'task_type': 'trend_detection',
            'keyword_trends_found': len(keyword_trends),
            'emerging_topics_found': len(emerging_topics),
            'cross_domain_trends_found': len(cross_domain_trends),
            'top_trending_keywords': [t.keyword for t in keyword_trends[:5]],
            'top_emerging_topics': [t['topic'] for t in emerging_topics[:5]]
        }
    
    async def _run_correlation_analysis_task(self, task: BatchTask) -> Dict:
        """Run correlation analysis task."""
        params = task.parameters
        
        loop = asyncio.get_event_loop()
        
        # Run paper correlations
        correlations = await loop.run_in_executor(
            None,
            self.correlation_analyzer.find_paper_correlations,
            params.get('paper_ids'),
            params.get('correlation_threshold', 0.3),
            params.get('max_correlations', 200)
        )
        
        # Run clustering if requested
        clusters = []
        if params.get('include_clustering', False):
            clusters = await loop.run_in_executor(
                None,
                self.correlation_analyzer.cluster_papers_by_topic,
                90,  # timeframe_days
                3    # min_cluster_size
            )
        
        return {
            'task_type': 'correlation_analysis',
            'correlations_found': len(correlations),
            'clusters_found': len(clusters),
            'strongest_correlations': [
                {
                    'paper1': c.paper1_id,
                    'paper2': c.paper2_id,
                    'score': c.correlation_score,
                    'type': c.correlation_type
                }
                for c in correlations[:5]
            ],
            'largest_clusters': [
                {
                    'topic': c.primary_topic,
                    'papers_count': len(c.paper_ids),
                    'coherence': c.coherence_score
                }
                for c in clusters[:3]
            ]
        }
    
    async def _wait_for_task_completion(self):
        """Wait for at least one running task to complete."""
        while len(self.running_tasks) >= self.max_concurrent_tasks:
            await asyncio.sleep(1)
    
    def _get_next_batch_window(self) -> datetime:
        """Get the next overnight batch processing window."""
        now = datetime.now()
        
        # Default batch window: 2 AM - 6 AM
        batch_start_hour = 2
        
        # If it's before 2 AM today, schedule for 2 AM today
        # Otherwise, schedule for 2 AM tomorrow
        if now.hour < batch_start_hour:
            next_batch = now.replace(hour=batch_start_hour, minute=0, second=0, microsecond=0)
        else:
            next_batch = (now + timedelta(days=1)).replace(
                hour=batch_start_hour, minute=0, second=0, microsecond=0
            )
        
        return next_batch
    
    def _compile_batch_results(self, tasks: List[BatchTask]) -> Dict[str, Any]:
        """Compile summary results from completed tasks."""
        results = {
            'nlp_analysis': {
                'total_papers_processed': 0,
                'successful_analyses': 0,
                'failed_analyses': 0
            },
            'trend_detection': {
                'keyword_trends_found': 0,
                'emerging_topics_found': 0,
                'cross_domain_trends_found': 0,
                'top_keywords': []
            },
            'correlation_analysis': {
                'correlations_found': 0,
                'clusters_found': 0,
                'strongest_correlations': []
            }
        }
        
        for task in tasks:
            if task.status == 'completed' and task.result:
                task_type = task.result.get('task_type')
                
                if task_type == 'nlp_analysis':
                    results['nlp_analysis']['total_papers_processed'] += task.result.get('papers_processed', 0)
                    results['nlp_analysis']['successful_analyses'] += task.result.get('successful_analyses', 0)
                    results['nlp_analysis']['failed_analyses'] += task.result.get('failed_analyses', 0)
                
                elif task_type == 'trend_detection':
                    results['trend_detection']['keyword_trends_found'] += task.result.get('keyword_trends_found', 0)
                    results['trend_detection']['emerging_topics_found'] += task.result.get('emerging_topics_found', 0)
                    results['trend_detection']['cross_domain_trends_found'] += task.result.get('cross_domain_trends_found', 0)
                    
                    # Collect top keywords
                    top_keywords = task.result.get('top_trending_keywords', [])
                    results['trend_detection']['top_keywords'].extend(top_keywords)
                
                elif task_type == 'correlation_analysis':
                    results['correlation_analysis']['correlations_found'] += task.result.get('correlations_found', 0)
                    results['correlation_analysis']['clusters_found'] += task.result.get('clusters_found', 0)
                    
                    # Collect strongest correlations
                    strong_corrs = task.result.get('strongest_correlations', [])
                    results['correlation_analysis']['strongest_correlations'].extend(strong_corrs)
        
        # Deduplicate and limit top results
        results['trend_detection']['top_keywords'] = list(set(results['trend_detection']['top_keywords']))[:10]
        results['correlation_analysis']['strongest_correlations'] = results['correlation_analysis']['strongest_correlations'][:10]
        
        return results
    
    def get_task_status(self, task_id: str) -> Optional[Dict]:
        """Get status of a specific task."""
        for task in self.task_queue:
            if task.task_id == task_id:
                return {
                    'task_id': task.task_id,
                    'task_type': task.task_type,
                    'status': task.status,
                    'created_at': task.created_at.isoformat(),
                    'scheduled_for': task.scheduled_for.isoformat(),
                    'priority': task.priority,
                    'error_message': task.error_message
                }
        return None
    
    def get_queue_status(self) -> Dict:
        """Get overall queue status."""
        pending = len([t for t in self.task_queue if t.status == 'pending'])
        running = len(self.running_tasks)
        completed = len([t for t in self.task_queue if t.status == 'completed'])
        failed = len([t for t in self.task_queue if t.status == 'failed'])
        
        return {
            'total_tasks': len(self.task_queue),
            'pending_tasks': pending,
            'running_tasks': running,
            'completed_tasks': completed,
            'failed_tasks': failed,
            'next_batch_window': self._get_next_batch_window().isoformat()
        }