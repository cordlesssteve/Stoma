"""Scheduler manager for controlling and monitoring scheduled tasks."""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional

from .base import CronScheduler, ScheduledTask, TaskExecution, TaskStatus, TaskPriority
from .config import load_schedule_config, create_custom_task
from ..storage.base import PostgreSQLStorage
from ..config.settings import load_config

logger = logging.getLogger(__name__)


class SchedulerManager:
    """High-level scheduler management interface."""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        self.scheduler: Optional[CronScheduler] = None
        self.storage: Optional[PostgreSQLStorage] = None
        self.state_file = Path("scheduler_state.json")
        self.is_running = False
    
    async def initialize(self) -> None:
        """Initialize scheduler and storage connections."""
        logger.info("Initializing SchedulerManager")
        
        # Initialize storage if configured
        if "storage" in self.config and self.config["storage"]["type"] == "postgresql":
            self.storage = PostgreSQLStorage(self.config["storage"])
            await self.storage.connect()
            logger.info("✓ Storage connected")
        
        # Initialize scheduler
        scheduler_config = self.config.get("scheduler", {})
        scheduler_config.update({
            "check_interval_seconds": scheduler_config.get("check_interval_seconds", 60),
            "max_concurrent_tasks": scheduler_config.get("max_concurrent_tasks", 3)
        })
        
        self.scheduler = CronScheduler(scheduler_config)
        
        # Load saved state
        await self._load_state()
        
        # Load default tasks if none exist
        if not self.scheduler.tasks:
            await self._load_default_tasks()
        
        logger.info("✓ SchedulerManager initialized")
    
    async def start(self) -> None:
        """Start the scheduler."""
        if not self.scheduler:
            raise RuntimeError("Scheduler not initialized. Call initialize() first.")
        
        logger.info("Starting scheduler")
        self.is_running = True
        
        # Configure storage for all tasks
        await self._configure_task_storage()
        
        await self.scheduler.start()
        logger.info("✓ Scheduler started")
    
    async def stop(self) -> None:
        """Stop the scheduler."""
        if self.scheduler:
            logger.info("Stopping scheduler")
            await self.scheduler.stop()
            self.is_running = False
            
            # Save current state
            await self._save_state()
            logger.info("✓ Scheduler stopped")
        
        if self.storage:
            await self.storage.disconnect()
            logger.info("✓ Storage disconnected")
    
    async def add_task(self, task: ScheduledTask) -> None:
        """Add a new scheduled task."""
        if not self.scheduler:
            raise RuntimeError("Scheduler not initialized")
        
        await self.scheduler.add_task(task)
        await self._save_state()
        logger.info(f"Added task: {task.id}")
    
    async def remove_task(self, task_id: str) -> None:
        """Remove a scheduled task."""
        if not self.scheduler:
            raise RuntimeError("Scheduler not initialized")
        
        await self.scheduler.remove_task(task_id)
        await self._save_state()
        logger.info(f"Removed task: {task_id}")
    
    async def enable_task(self, task_id: str) -> None:
        """Enable a task."""
        task = self.get_task(task_id)
        if task:
            task.enabled = True
            if self.scheduler:
                task.next_run = self.scheduler._calculate_next_run(task)
            await self._save_state()
            logger.info(f"Enabled task: {task_id}")
    
    async def disable_task(self, task_id: str) -> None:
        """Disable a task."""
        task = self.get_task(task_id)
        if task:
            task.enabled = False
            task.next_run = None
            await self._save_state()
            logger.info(f"Disabled task: {task_id}")
    
    async def run_task_now(self, task_id: str) -> TaskExecution:
        """Execute a task immediately."""
        if not self.scheduler:
            raise RuntimeError("Scheduler not initialized")
        
        task = self.scheduler.get_task(task_id)
        if not task:
            raise ValueError(f"Task not found: {task_id}")
        
        logger.info(f"Running task immediately: {task_id}")
        return await self.scheduler.execute_task(task)
    
    def get_task(self, task_id: str) -> Optional[ScheduledTask]:
        """Get a task by ID."""
        if not self.scheduler:
            return None
        return self.scheduler.get_task(task_id)
    
    def get_all_tasks(self) -> List[ScheduledTask]:
        """Get all scheduled tasks."""
        if not self.scheduler:
            return []
        return self.scheduler.get_all_tasks()
    
    def get_task_history(self, task_id: Optional[str] = None, limit: int = 100) -> List[TaskExecution]:
        """Get task execution history."""
        if not self.scheduler:
            return []
        return self.scheduler.get_task_history(task_id, limit)
    
    def get_status(self) -> Dict[str, Any]:
        """Get scheduler status and statistics."""
        if not self.scheduler:
            return {"status": "not_initialized"}
        
        tasks = self.scheduler.get_all_tasks()
        running_tasks = [t for t in tasks if t.status == TaskStatus.RUNNING]
        enabled_tasks = [t for t in tasks if t.enabled]
        
        # Next run times
        next_runs = []
        for task in enabled_tasks:
            if task.next_run:
                next_runs.append({
                    "task_id": task.id,
                    "task_name": task.name,
                    "next_run": task.next_run.isoformat(),
                    "priority": task.priority.name
                })
        
        next_runs.sort(key=lambda x: x["next_run"])
        
        # Recent executions
        recent_history = self.get_task_history(limit=10)
        
        return {
            "status": "running" if self.is_running else "stopped",
            "total_tasks": len(tasks),
            "enabled_tasks": len(enabled_tasks),
            "running_tasks": len(running_tasks),
            "next_runs": next_runs[:5],  # Next 5 scheduled runs
            "recent_executions": [
                {
                    "task_id": exec.task_id,
                    "started_at": exec.started_at.isoformat(),
                    "status": exec.status.value,
                    "results_count": exec.results_count,
                    "duration_seconds": exec.duration_seconds
                }
                for exec in recent_history
            ]
        }
    
    async def load_preset_config(self, config_type: str) -> None:
        """Load a preset configuration of tasks."""
        logger.info(f"Loading preset config: {config_type}")
        
        try:
            preset_tasks = load_schedule_config(config_type)
            
            if self.scheduler:
                # Clear existing tasks
                existing_task_ids = list(self.scheduler.tasks.keys())
                for task_id in existing_task_ids:
                    await self.scheduler.remove_task(task_id)
                
                # Add preset tasks
                for task in preset_tasks:
                    await self.scheduler.add_task(task)
                
                await self._save_state()
            
            logger.info(f"✓ Loaded {len(preset_tasks)} tasks from preset: {config_type}")
            
        except Exception as e:
            logger.error(f"Failed to load preset config {config_type}: {e}")
            raise
    
    async def create_and_add_task(
        self,
        task_id: str,
        name: str,
        collector_type: str,
        collector_config: Dict[str, Any],
        schedule_pattern: str,
        priority: str = "normal",
        enabled: bool = True
    ) -> None:
        """Create and add a custom task."""
        
        priority_map = {
            "low": TaskPriority.LOW,
            "normal": TaskPriority.NORMAL,
            "high": TaskPriority.HIGH,
            "critical": TaskPriority.CRITICAL
        }
        
        task = create_custom_task(
            task_id=task_id,
            name=name,
            collector_type=collector_type,
            collector_config=collector_config,
            schedule_pattern=schedule_pattern,
            priority=priority_map.get(priority.lower(), TaskPriority.NORMAL),
            enabled=enabled
        )
        
        await self.add_task(task)
    
    async def _load_default_tasks(self) -> None:
        """Load default task configuration."""
        default_config = self.config.get("scheduler", {}).get("default_preset", "default")
        await self.load_preset_config(default_config)
    
    async def _configure_task_storage(self) -> None:
        """Configure storage for all tasks."""
        if not self.storage or not self.scheduler:
            return
        
        # Add storage reference to scheduler for task execution
        self.scheduler.storage = self.storage
    
    async def _save_state(self) -> None:
        """Save scheduler state to file."""
        if not self.scheduler:
            return
        
        try:
            state = {
                "tasks": [],
                "last_saved": datetime.now().isoformat()
            }
            
            for task in self.scheduler.get_all_tasks():
                task_data = {
                    "id": task.id,
                    "name": task.name,
                    "collector_type": task.collector_type,
                    "config": task.config,
                    "schedule_pattern": task.schedule_pattern,
                    "priority": task.priority.value,
                    "timeout_seconds": task.timeout_seconds,
                    "retry_count": task.retry_count,
                    "enabled": task.enabled,
                    "last_run": task.last_run.isoformat() if task.last_run else None,
                    "next_run": task.next_run.isoformat() if task.next_run else None,
                    "status": task.status.value,
                    "error_message": task.error_message
                }
                state["tasks"].append(task_data)
            
            with open(self.state_file, "w") as f:
                json.dump(state, f, indent=2)
            
        except Exception as e:
            logger.error(f"Failed to save scheduler state: {e}")
    
    async def _load_state(self) -> None:
        """Load scheduler state from file."""
        if not self.state_file.exists() or not self.scheduler:
            return
        
        try:
            with open(self.state_file, "r") as f:
                state = json.load(f)
            
            for task_data in state.get("tasks", []):
                task = ScheduledTask(
                    id=task_data["id"],
                    name=task_data["name"],
                    collector_type=task_data["collector_type"],
                    config=task_data["config"],
                    schedule_pattern=task_data["schedule_pattern"],
                    priority=TaskPriority(task_data["priority"]),
                    timeout_seconds=task_data["timeout_seconds"],
                    retry_count=task_data["retry_count"],
                    enabled=task_data["enabled"],
                    last_run=datetime.fromisoformat(task_data["last_run"]) if task_data["last_run"] else None,
                    next_run=datetime.fromisoformat(task_data["next_run"]) if task_data["next_run"] else None,
                    status=TaskStatus(task_data["status"]),
                    error_message=task_data["error_message"]
                )
                
                await self.scheduler.add_task(task)
            
            logger.info(f"Loaded {len(state.get('tasks', []))} tasks from saved state")
            
        except Exception as e:
            logger.error(f"Failed to load scheduler state: {e}")


async def run_scheduler_daemon(config_path: str = "config.yaml", preset: str = "default"):
    """Run scheduler as a daemon process."""
    manager = SchedulerManager(config_path)
    
    try:
        await manager.initialize()
        
        # Load preset if specified
        if preset != "default":
            await manager.load_preset_config(preset)
        
        await manager.start()
        
        # Keep running until interrupted
        try:
            while True:
                await asyncio.sleep(60)
                
                # Log status periodically
                status = manager.get_status()
                logger.info(f"Scheduler status: {status['running_tasks']} running, "
                          f"{status['enabled_tasks']} enabled, next run in schedule")
                
        except KeyboardInterrupt:
            logger.info("Shutdown requested")
    
    finally:
        await manager.stop()


if __name__ == "__main__":
    import sys
    
    preset = sys.argv[1] if len(sys.argv) > 1 else "default"
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    asyncio.run(run_scheduler_daemon(preset=preset))