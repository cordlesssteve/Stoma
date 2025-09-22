"""Base scheduler classes and task management."""

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Awaitable
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class ScheduledTask:
    """A scheduled task definition."""
    id: str
    name: str
    collector_type: str
    config: Dict[str, Any]
    schedule_pattern: str  # cron-like pattern
    priority: TaskPriority = TaskPriority.NORMAL
    timeout_seconds: int = 3600
    retry_count: int = 3
    enabled: bool = True
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    status: TaskStatus = TaskStatus.PENDING
    error_message: Optional[str] = None


@dataclass
class TaskExecution:
    """A task execution record."""
    task_id: str
    execution_id: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    status: TaskStatus = TaskStatus.RUNNING
    results_count: int = 0
    error_message: Optional[str] = None
    duration_seconds: Optional[float] = None


class BaseScheduler(ABC):
    """Abstract base scheduler."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.tasks: Dict[str, ScheduledTask] = {}
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self.task_history: List[TaskExecution] = []
        self.is_running = False
        self._shutdown_event = asyncio.Event()
    
    @abstractmethod
    async def start(self) -> None:
        """Start the scheduler."""
        pass
    
    @abstractmethod
    async def stop(self) -> None:
        """Stop the scheduler."""
        pass
    
    @abstractmethod
    async def add_task(self, task: ScheduledTask) -> None:
        """Add a task to the schedule."""
        pass
    
    @abstractmethod
    async def remove_task(self, task_id: str) -> None:
        """Remove a task from the schedule."""
        pass
    
    @abstractmethod
    async def execute_task(self, task: ScheduledTask) -> TaskExecution:
        """Execute a single task."""
        pass
    
    def get_task(self, task_id: str) -> Optional[ScheduledTask]:
        """Get a task by ID."""
        return self.tasks.get(task_id)
    
    def get_all_tasks(self) -> List[ScheduledTask]:
        """Get all scheduled tasks."""
        return list(self.tasks.values())
    
    def get_task_history(self, task_id: Optional[str] = None, limit: int = 100) -> List[TaskExecution]:
        """Get task execution history."""
        history = self.task_history
        if task_id:
            history = [exec for exec in history if exec.task_id == task_id]
        return history[-limit:]


class CronScheduler(BaseScheduler):
    """Cron-like scheduler implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.check_interval = config.get("check_interval_seconds", 60)
        self.max_concurrent_tasks = config.get("max_concurrent_tasks", 5)
    
    async def start(self) -> None:
        """Start the scheduler loop."""
        logger.info("Starting CronScheduler")
        self.is_running = True
        
        # Calculate next run times for all tasks
        for task in self.tasks.values():
            if task.enabled:
                task.next_run = self._calculate_next_run(task)
        
        # Start the main scheduler loop
        asyncio.create_task(self._scheduler_loop())
    
    async def stop(self) -> None:
        """Stop the scheduler."""
        logger.info("Stopping CronScheduler")
        self.is_running = False
        self._shutdown_event.set()
        
        # Cancel all running tasks
        for task_id, running_task in self.running_tasks.items():
            logger.info(f"Cancelling running task: {task_id}")
            running_task.cancel()
        
        # Wait for tasks to finish
        if self.running_tasks:
            await asyncio.gather(*self.running_tasks.values(), return_exceptions=True)
    
    async def add_task(self, task: ScheduledTask) -> None:
        """Add a task to the schedule."""
        logger.info(f"Adding task: {task.id} - {task.name}")
        
        if task.enabled:
            task.next_run = self._calculate_next_run(task)
        
        self.tasks[task.id] = task
    
    async def remove_task(self, task_id: str) -> None:
        """Remove a task from the schedule."""
        logger.info(f"Removing task: {task_id}")
        
        # Cancel if currently running
        if task_id in self.running_tasks:
            self.running_tasks[task_id].cancel()
            del self.running_tasks[task_id]
        
        # Remove from schedule
        if task_id in self.tasks:
            del self.tasks[task_id]
    
    async def execute_task(self, task: ScheduledTask) -> TaskExecution:
        """Execute a single task."""
        execution_id = f"{task.id}_{datetime.now().isoformat()}"
        execution = TaskExecution(
            task_id=task.id,
            execution_id=execution_id,
            started_at=datetime.now()
        )
        
        logger.info(f"Executing task: {task.id} - {task.name}")
        
        try:
            # Import and create collector dynamically
            collector_class = await self._get_collector_class(task.collector_type)
            collector = collector_class(task.config)
            
            # Execute collection
            results_count = 0
            async for result in collector.collect():
                if result.success:
                    results_count += 1
                    
                    # Store result if storage is configured
                    if hasattr(collector, 'storage') and collector.storage:
                        normalizer = await self._get_normalizer(result.source_type)
                        if normalizer:
                            normalized = await normalizer.normalize(result)
                            await collector.storage.store(normalized)
            
            # Update execution record
            execution.completed_at = datetime.now()
            execution.status = TaskStatus.COMPLETED
            execution.results_count = results_count
            execution.duration_seconds = (execution.completed_at - execution.started_at).total_seconds()
            
            # Update task record
            task.last_run = execution.started_at
            task.next_run = self._calculate_next_run(task)
            task.status = TaskStatus.COMPLETED
            task.error_message = None
            
            logger.info(f"Task completed: {task.id} - collected {results_count} items")
            
        except Exception as e:
            execution.completed_at = datetime.now()
            execution.status = TaskStatus.FAILED
            execution.error_message = str(e)
            execution.duration_seconds = (execution.completed_at - execution.started_at).total_seconds()
            
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            
            logger.error(f"Task failed: {task.id} - {e}")
        
        # Store execution history
        self.task_history.append(execution)
        
        # Cleanup old history (keep last 1000 executions)
        if len(self.task_history) > 1000:
            self.task_history = self.task_history[-1000:]
        
        return execution
    
    async def _scheduler_loop(self) -> None:
        """Main scheduler loop."""
        while self.is_running:
            try:
                await self._check_and_execute_tasks()
                
                # Wait for next check or shutdown
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(), 
                        timeout=self.check_interval
                    )
                    break  # Shutdown requested
                except asyncio.TimeoutError:
                    continue  # Normal check interval
                    
            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")
                await asyncio.sleep(self.check_interval)
    
    async def _check_and_execute_tasks(self) -> None:
        """Check for tasks that need to run and execute them."""
        now = datetime.now()
        
        # Clean up completed running tasks
        completed_tasks = []
        for task_id, running_task in self.running_tasks.items():
            if running_task.done():
                completed_tasks.append(task_id)
        
        for task_id in completed_tasks:
            del self.running_tasks[task_id]
        
        # Check if we can run more tasks
        if len(self.running_tasks) >= self.max_concurrent_tasks:
            return
        
        # Find tasks that need to run
        ready_tasks = []
        for task in self.tasks.values():
            if (task.enabled and 
                task.next_run and 
                task.next_run <= now and 
                task.id not in self.running_tasks):
                ready_tasks.append(task)
        
        # Sort by priority and next_run time
        ready_tasks.sort(key=lambda t: (t.priority.value, t.next_run), reverse=True)
        
        # Execute tasks up to the concurrent limit
        slots_available = self.max_concurrent_tasks - len(self.running_tasks)
        for task in ready_tasks[:slots_available]:
            task.status = TaskStatus.RUNNING
            running_task = asyncio.create_task(self.execute_task(task))
            self.running_tasks[task.id] = running_task
    
    def _calculate_next_run(self, task: ScheduledTask) -> datetime:
        """Calculate next run time based on schedule pattern."""
        # Simplified cron parsing - supports basic patterns
        now = datetime.now()
        pattern = task.schedule_pattern
        
        # Simple patterns supported:
        # "hourly" - every hour
        # "daily" - every day at midnight
        # "weekly" - every week on Sunday
        # "daily_at_HH:MM" - daily at specific time
        # "interval_XXm" - every XX minutes
        # "interval_XXh" - every XX hours
        
        if pattern == "hourly":
            next_run = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        elif pattern == "daily":
            next_run = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        elif pattern == "weekly":
            days_until_sunday = (6 - now.weekday()) % 7
            if days_until_sunday == 0:
                days_until_sunday = 7
            next_run = (now + timedelta(days=days_until_sunday)).replace(hour=0, minute=0, second=0, microsecond=0)
        elif pattern.startswith("daily_at_"):
            time_part = pattern.split("_", 2)[2]  # Extract HH:MM
            try:
                hour, minute = map(int, time_part.split(":"))
                next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                if next_run <= now:
                    next_run += timedelta(days=1)
            except ValueError:
                next_run = now + timedelta(hours=1)  # Default fallback
        elif pattern.startswith("interval_"):
            interval_part = pattern.split("_", 1)[1]
            if interval_part.endswith("m"):
                minutes = int(interval_part[:-1])
                next_run = now + timedelta(minutes=minutes)
            elif interval_part.endswith("h"):
                hours = int(interval_part[:-1])
                next_run = now + timedelta(hours=hours)
            else:
                next_run = now + timedelta(hours=1)  # Default fallback
        else:
            # Default: run every hour
            next_run = now + timedelta(hours=1)
        
        return next_run
    
    async def _get_collector_class(self, collector_type: str):
        """Dynamically import and return collector class."""
        from ..collectors.arxiv import ArXivCollector
        from ..collectors.github import GitHubCollector
        from ..collectors.sec_edgar import SECEdgarCollector
        
        collectors = {
            "arxiv": ArXivCollector,
            "github": GitHubCollector,
            "sec_edgar": SECEdgarCollector
        }
        
        if collector_type not in collectors:
            raise ValueError(f"Unknown collector type: {collector_type}")
        
        return collectors[collector_type]
    
    async def _get_normalizer(self, source_type):
        """Get appropriate normalizer for source type."""
        from ..normalizers.base import AcademicNormalizer
        from ..normalizers.code_projects import CodeProjectsNormalizer
        from ..normalizers.public_docs import PublicDocsNormalizer
        from ..collectors.base import SourceType
        
        normalizers = {
            SourceType.ACADEMIC: AcademicNormalizer({}),
            SourceType.CODE_PROJECTS: CodeProjectsNormalizer({}),
            SourceType.PUBLIC_DOCS: PublicDocsNormalizer({})
        }
        
        return normalizers.get(source_type)