"""Scheduler configuration and task definitions."""

from typing import Dict, List, Any
from datetime import datetime

from .base import ScheduledTask, TaskPriority


class DefaultScheduleConfig:
    """Default scheduled tasks configuration."""
    
    @staticmethod
    def get_default_tasks() -> List[ScheduledTask]:
        """Get default scheduled tasks for common collection scenarios."""
        
        tasks = [
            # Academic: ArXiv trending papers - daily
            ScheduledTask(
                id="arxiv_daily_trending",
                name="ArXiv Daily Trending Papers",
                collector_type="arxiv",
                config={
                    "search_query": "cat:cs.AI OR cat:cs.LG OR cat:cs.CV",
                    "max_results": 50,
                    "sort_by": "submittedDate",
                    "sort_order": "descending"
                },
                schedule_pattern="daily_at_09:00",
                priority=TaskPriority.HIGH,
                timeout_seconds=1800
            ),
            
            # Academic: ArXiv ML papers - twice daily
            ScheduledTask(
                id="arxiv_ml_updates",
                name="ArXiv Machine Learning Updates",
                collector_type="arxiv",
                config={
                    "search_query": "cat:cs.LG OR cat:stat.ML",
                    "max_results": 30,
                    "sort_by": "submittedDate",
                    "sort_order": "descending"
                },
                schedule_pattern="interval_12h",
                priority=TaskPriority.NORMAL,
                timeout_seconds=1200
            ),
            
            # Code Projects: GitHub trending repos - daily
            ScheduledTask(
                id="github_trending_daily",
                name="GitHub Trending Repositories",
                collector_type="github",
                config={
                    "language": "",  # All languages
                    "since": "daily",
                    "limit": 50
                },
                schedule_pattern="daily_at_10:00",
                priority=TaskPriority.HIGH,
                timeout_seconds=2400
            ),
            
            # Code Projects: Python trending - daily
            ScheduledTask(
                id="github_python_trending",
                name="GitHub Python Trending",
                collector_type="github",
                config={
                    "language": "python",
                    "since": "daily",
                    "limit": 30
                },
                schedule_pattern="daily_at_11:00",
                priority=TaskPriority.NORMAL,
                timeout_seconds=1800
            ),
            
            # Code Projects: JavaScript trending - daily
            ScheduledTask(
                id="github_javascript_trending",
                name="GitHub JavaScript Trending",
                collector_type="github",
                config={
                    "language": "javascript",
                    "since": "daily",
                    "limit": 30
                },
                schedule_pattern="daily_at_11:30",
                priority=TaskPriority.NORMAL,
                timeout_seconds=1800
            ),
            
            # SEC: Recent filings - hourly during market hours
            ScheduledTask(
                id="sec_recent_filings",
                name="SEC Recent Filings",
                collector_type="sec_edgar",
                config={
                    "days_back": 1,
                    "filing_type": None  # All filing types
                },
                schedule_pattern="interval_2h",
                priority=TaskPriority.HIGH,
                timeout_seconds=3600
            ),
            
            # SEC: 8-K urgent filings - every 30 minutes during market hours
            ScheduledTask(
                id="sec_urgent_8k",
                name="SEC 8-K Urgent Filings",
                collector_type="sec_edgar",
                config={
                    "days_back": 1,
                    "filing_type": "8-K"
                },
                schedule_pattern="interval_30m",
                priority=TaskPriority.CRITICAL,
                timeout_seconds=1800
            ),
            
            # Weekly comprehensive scans
            ScheduledTask(
                id="arxiv_weekly_comprehensive",
                name="ArXiv Weekly Comprehensive Scan",
                collector_type="arxiv",
                config={
                    "search_query": "cat:cs.* OR cat:stat.* OR cat:math.*",
                    "max_results": 200,
                    "sort_by": "submittedDate",
                    "sort_order": "descending"
                },
                schedule_pattern="weekly",
                priority=TaskPriority.LOW,
                timeout_seconds=7200
            ),
            
            ScheduledTask(
                id="github_weekly_comprehensive",
                name="GitHub Weekly Comprehensive Scan",
                collector_type="github",
                config={
                    "language": "",
                    "since": "weekly",
                    "limit": 100
                },
                schedule_pattern="weekly",
                priority=TaskPriority.LOW,
                timeout_seconds=7200
            )
        ]
        
        return tasks
    
    @staticmethod
    def get_research_focused_tasks() -> List[ScheduledTask]:
        """Get tasks focused on research and academic content."""
        
        tasks = [
            ScheduledTask(
                id="arxiv_ai_research",
                name="ArXiv AI Research Papers",
                collector_type="arxiv",
                config={
                    "search_query": "cat:cs.AI",
                    "max_results": 25,
                    "sort_by": "submittedDate",
                    "sort_order": "descending"
                },
                schedule_pattern="daily_at_08:00",
                priority=TaskPriority.HIGH
            ),
            
            ScheduledTask(
                id="arxiv_nlp_research",
                name="ArXiv NLP Research Papers",
                collector_type="arxiv",
                config={
                    "search_query": "cat:cs.CL",
                    "max_results": 20,
                    "sort_by": "submittedDate",
                    "sort_order": "descending"
                },
                schedule_pattern="daily_at_08:30",
                priority=TaskPriority.HIGH
            ),
            
            ScheduledTask(
                id="arxiv_computer_vision",
                name="ArXiv Computer Vision Papers",
                collector_type="arxiv",
                config={
                    "search_query": "cat:cs.CV",
                    "max_results": 20,
                    "sort_by": "submittedDate",
                    "sort_order": "descending"
                },
                schedule_pattern="daily_at_09:00",
                priority=TaskPriority.HIGH
            )
        ]
        
        return tasks
    
    @staticmethod
    def get_business_focused_tasks() -> List[ScheduledTask]:
        """Get tasks focused on business and market intelligence."""
        
        tasks = [
            ScheduledTask(
                id="sec_10k_annual_reports",
                name="SEC 10-K Annual Reports",
                collector_type="sec_edgar",
                config={
                    "days_back": 1,
                    "filing_type": "10-K"
                },
                schedule_pattern="daily_at_07:00",
                priority=TaskPriority.HIGH
            ),
            
            ScheduledTask(
                id="sec_10q_quarterly_reports",
                name="SEC 10-Q Quarterly Reports",
                collector_type="sec_edgar",
                config={
                    "days_back": 1,
                    "filing_type": "10-Q"
                },
                schedule_pattern="daily_at_07:30",
                priority=TaskPriority.HIGH
            ),
            
            ScheduledTask(
                id="sec_insider_trading",
                name="SEC Insider Trading Reports",
                collector_type="sec_edgar",
                config={
                    "days_back": 1,
                    "filing_type": "4"
                },
                schedule_pattern="interval_4h",
                priority=TaskPriority.HIGH
            )
        ]
        
        return tasks
    
    @staticmethod
    def get_development_focused_tasks() -> List[ScheduledTask]:
        """Get tasks focused on software development and technology."""
        
        tasks = [
            ScheduledTask(
                id="github_python_daily",
                name="GitHub Python Daily Trending",
                collector_type="github",
                config={
                    "language": "python",
                    "since": "daily",
                    "limit": 25
                },
                schedule_pattern="daily_at_10:00",
                priority=TaskPriority.HIGH
            ),
            
            ScheduledTask(
                id="github_typescript_daily",
                name="GitHub TypeScript Daily Trending",
                collector_type="github",
                config={
                    "language": "typescript",
                    "since": "daily",
                    "limit": 20
                },
                schedule_pattern="daily_at_10:30",
                priority=TaskPriority.NORMAL
            ),
            
            ScheduledTask(
                id="github_rust_daily",
                name="GitHub Rust Daily Trending",
                collector_type="github",
                config={
                    "language": "rust",
                    "since": "daily",
                    "limit": 15
                },
                schedule_pattern="daily_at_11:00",
                priority=TaskPriority.NORMAL
            ),
            
            ScheduledTask(
                id="github_go_daily",
                name="GitHub Go Daily Trending",
                collector_type="github",
                config={
                    "language": "go",
                    "since": "daily",
                    "limit": 15
                },
                schedule_pattern="daily_at_11:30",
                priority=TaskPriority.NORMAL
            )
        ]
        
        return tasks


def load_schedule_config(config_type: str = "default") -> List[ScheduledTask]:
    """Load scheduled tasks based on configuration type."""
    
    if config_type == "research":
        return DefaultScheduleConfig.get_research_focused_tasks()
    elif config_type == "business":
        return DefaultScheduleConfig.get_business_focused_tasks()
    elif config_type == "development":
        return DefaultScheduleConfig.get_development_focused_tasks()
    elif config_type == "default":
        return DefaultScheduleConfig.get_default_tasks()
    else:
        raise ValueError(f"Unknown config type: {config_type}")


def create_custom_task(
    task_id: str,
    name: str,
    collector_type: str,
    collector_config: Dict[str, Any],
    schedule_pattern: str,
    priority: TaskPriority = TaskPriority.NORMAL,
    timeout_seconds: int = 3600,
    enabled: bool = True
) -> ScheduledTask:
    """Create a custom scheduled task."""
    
    return ScheduledTask(
        id=task_id,
        name=name,
        collector_type=collector_type,
        config=collector_config,
        schedule_pattern=schedule_pattern,
        priority=priority,
        timeout_seconds=timeout_seconds,
        enabled=enabled
    )