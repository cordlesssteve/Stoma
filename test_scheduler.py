#!/usr/bin/env python3
"""Test the KnowHunt scheduler system."""

import asyncio
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from knowhunt.scheduler.manager import SchedulerManager
from knowhunt.scheduler.config import create_custom_task, DefaultScheduleConfig
from knowhunt.scheduler.base import TaskPriority


async def test_scheduler_basic():
    """Test basic scheduler functionality."""
    print("🧪 Testing KnowHunt Scheduler System")
    print("=" * 50)
    
    # Initialize scheduler manager
    print("\n1. Initializing Scheduler Manager...")
    manager = SchedulerManager()
    
    try:
        await manager.initialize()
        print("✓ Scheduler manager initialized")
        
        # Test task creation
        print("\n2. Creating test task...")
        test_task = create_custom_task(
            task_id="test_arxiv_quick",
            name="Test ArXiv Quick Scan",
            collector_type="arxiv",
            collector_config={
                "search_query": "machine learning",
                "max_results": 3,
                "sort_by": "submittedDate",
                "sort_order": "descending"
            },
            schedule_pattern="interval_5m",  # Every 5 minutes
            priority=TaskPriority.HIGH
        )
        
        await manager.add_task(test_task)
        print(f"✓ Created task: {test_task.name}")
        
        # List tasks
        print("\n3. Listing scheduled tasks...")
        tasks = manager.get_all_tasks()
        for task in tasks:
            print(f"  - {task.name} ({task.collector_type}) - {task.schedule_pattern}")
        
        # Test immediate execution
        print("\n4. Testing immediate task execution...")
        if tasks:
            task_to_run = tasks[0]
            print(f"Running task: {task_to_run.name}")
            
            execution = await manager.run_task_now(task_to_run.id)
            
            print(f"✓ Task execution completed")
            print(f"  Status: {execution.status.value}")
            print(f"  Results: {execution.results_count}")
            print(f"  Duration: {execution.duration_seconds:.2f}s")
            
            if execution.error_message:
                print(f"  Error: {execution.error_message}")
        
        # Get status
        print("\n5. Scheduler status...")
        status = manager.get_status()
        print(f"  Status: {status['status']}")
        print(f"  Total tasks: {status['total_tasks']}")
        print(f"  Enabled tasks: {status['enabled_tasks']}")
        
        if status['next_runs']:
            print(f"  Next run: {status['next_runs'][0]['task_name']} at {status['next_runs'][0]['next_run']}")
        
        # Test history
        print("\n6. Execution history...")
        history = manager.get_task_history(limit=5)
        for exec in history:
            print(f"  - {exec.task_id}: {exec.status.value} ({exec.results_count} results)")
        
        print("\n✅ Basic scheduler tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Scheduler test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await manager.stop()
        print("✓ Scheduler manager stopped")


async def test_preset_configs():
    """Test preset configurations."""
    print("\n🎛️ Testing Preset Configurations")
    print("=" * 40)
    
    presets = ["research", "business", "development"]
    
    for preset in presets:
        print(f"\n Testing {preset} preset...")
        
        try:
            tasks = DefaultScheduleConfig.get_research_focused_tasks() if preset == "research" else \
                    DefaultScheduleConfig.get_business_focused_tasks() if preset == "business" else \
                    DefaultScheduleConfig.get_development_focused_tasks()
            
            print(f"✓ {preset.title()} preset: {len(tasks)} tasks")
            for task in tasks[:3]:  # Show first 3 tasks
                print(f"  - {task.name} ({task.schedule_pattern})")
            
        except Exception as e:
            print(f"❌ Failed to load {preset} preset: {e}")


async def main():
    """Run all scheduler tests."""
    print("🔍 KnowHunt Scheduler Test Suite")
    print("=" * 60)
    
    await test_scheduler_basic()
    await test_preset_configs()
    
    print("\n🎉 All scheduler tests completed!")


if __name__ == "__main__":
    asyncio.run(main())