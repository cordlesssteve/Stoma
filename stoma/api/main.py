"""FastAPI web application for KnowHunt dashboard."""

import json
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from ..storage.base import PostgreSQLStorage, NormalizedData
from ..collectors.base import SourceType
from ..config.settings import load_config
from ..scheduler.manager import SchedulerManager


app = FastAPI(
    title="KnowHunt Research Intelligence System",
    description="Web dashboard for exploring collected research data",
    version="0.1.0"
)

# Setup templates and static files
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

# Global instances
storage: Optional[PostgreSQLStorage] = None
scheduler_manager: Optional[SchedulerManager] = None


class SearchRequest(BaseModel):
    query: str
    source_types: Optional[List[str]] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    limit: int = 50


class DataSummary(BaseModel):
    total_items: int
    source_counts: Dict[str, int]
    recent_items: List[Dict[str, Any]]
    top_categories: List[Dict[str, Any]]


@app.on_event("startup")
async def startup_event():
    """Initialize storage and scheduler connections on startup."""
    global storage, scheduler_manager
    
    try:
        config = load_config("config.yaml")
        
        # Initialize storage
        if "storage" in config and config["storage"]["type"] == "postgresql":
            storage = PostgreSQLStorage(config["storage"])
            await storage.connect()
            print("✓ Connected to database")
        else:
            print("⚠ No PostgreSQL storage configured")
        
        # Initialize scheduler manager
        scheduler_manager = SchedulerManager("config.yaml")
        await scheduler_manager.initialize()
        print("✓ Scheduler manager initialized")
        
    except Exception as e:
        print(f"⚠ Failed to initialize services: {e}")
        storage = None
        scheduler_manager = None


@app.on_event("shutdown")
async def shutdown_event():
    """Close storage and scheduler connections on shutdown."""
    global storage, scheduler_manager
    
    if scheduler_manager:
        await scheduler_manager.stop()
        print("✓ Scheduler stopped")
    
    if storage:
        await storage.disconnect()
        print("✓ Disconnected from database")


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard page."""
    return templates.TemplateResponse("dashboard.html", {"request": request})


@app.get("/api/summary")
async def get_summary() -> DataSummary:
    """Get data summary for dashboard."""
    if not storage:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        # This is a simplified implementation
        # In a real system, you'd want proper aggregation queries
        
        # Get recent items as a sample
        recent_items = []
        async for item in storage.search("", limit=20):
            recent_items.append({
                "id": item.id,
                "title": item.title[:100] + "..." if len(item.title) > 100 else item.title,
                "source_type": item.source_type.value,
                "published_date": item.published_date.isoformat() if item.published_date else None,
                "url": item.url,
                "categories": item.categories[:3]  # First 3 categories
            })
        
        # Count by source type (simplified)
        source_counts = {}
        for item in recent_items:
            source_type = item["source_type"]
            source_counts[source_type] = source_counts.get(source_type, 0) + 1
        
        # Count categories (simplified)
        category_counts = {}
        for item in recent_items:
            for category in item["categories"]:
                category_counts[category] = category_counts.get(category, 0) + 1
        
        top_categories = [
            {"name": cat, "count": count} 
            for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        ]
        
        return DataSummary(
            total_items=len(recent_items),  # This would be a proper count in production
            source_counts=source_counts,
            recent_items=recent_items[:10],  # Latest 10 items
            top_categories=top_categories
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get summary: {e}")


@app.get("/api/search")
async def search_data(
    query: str = Query(..., description="Search query"),
    source_types: Optional[str] = Query(None, description="Comma-separated source types"),
    date_from: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    date_to: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    limit: int = Query(50, description="Maximum results")
) -> List[Dict[str, Any]]:
    """Search collected data."""
    if not storage:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        # Parse source types
        source_type_list = None
        if source_types:
            source_type_list = [
                SourceType(st.strip()) 
                for st in source_types.split(",") 
                if st.strip() in [e.value for e in SourceType]
            ]
        
        # Parse dates
        date_from_dt = None
        date_to_dt = None
        
        if date_from:
            try:
                date_from_dt = datetime.fromisoformat(date_from)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date_from format")
        
        if date_to:
            try:
                date_to_dt = datetime.fromisoformat(date_to)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date_to format")
        
        # Perform search
        results = []
        async for item in storage.search(
            query=query,
            source_types=source_type_list,
            date_from=date_from_dt,
            date_to=date_to_dt,
            limit=limit
        ):
            results.append({
                "id": item.id,
                "title": item.title,
                "content": item.content[:500] + "..." if len(item.content) > 500 else item.content,
                "summary": item.summary,
                "authors": item.authors,
                "source_type": item.source_type.value,
                "published_date": item.published_date.isoformat() if item.published_date else None,
                "collected_date": item.collected_date.isoformat() if item.collected_date else None,
                "url": item.url,
                "keywords": item.keywords,
                "categories": item.categories,
                "tags": item.tags,
                "metrics": item.metrics
            })
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")


@app.get("/api/categories")
async def get_categories() -> List[Dict[str, Any]]:
    """Get all categories with counts."""
    if not storage:
        raise HTTPException(status_code=503, detail="Database not available")
    
    # This is a simplified implementation
    # In production, you'd want a proper aggregation query
    try:
        category_counts = {}
        
        # Sample recent items to get categories
        async for item in storage.search("", limit=100):
            for category in item.categories:
                category_counts[category] = category_counts.get(category, 0) + 1
        
        categories = [
            {"name": cat, "count": count}
            for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
        ]
        
        return categories
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get categories: {e}")


@app.get("/api/source-types")
async def get_source_types() -> List[Dict[str, str]]:
    """Get available source types."""
    return [
        {"value": source_type.value, "label": source_type.value.replace("_", " ").title()}
        for source_type in SourceType
    ]


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    health_status = {"status": "healthy"}
    
    # Check database
    if storage:
        try:
            healthy = await storage.health_check()
            if healthy:
                health_status["database"] = "connected"
            else:
                health_status["database"] = "connection_issues"
                health_status["status"] = "degraded"
        except Exception as e:
            health_status["database"] = f"error: {e}"
            health_status["status"] = "unhealthy"
    else:
        health_status["database"] = "not_configured"
        health_status["status"] = "degraded"
    
    # Check scheduler
    if scheduler_manager:
        scheduler_status = scheduler_manager.get_status()
        health_status["scheduler"] = scheduler_status["status"]
        health_status["scheduled_tasks"] = scheduler_status["total_tasks"]
    else:
        health_status["scheduler"] = "not_configured"
        health_status["scheduled_tasks"] = 0
    
    return health_status


# Scheduler API Endpoints
@app.get("/api/scheduler/status")
async def get_scheduler_status() -> Dict[str, Any]:
    """Get scheduler status and statistics."""
    if not scheduler_manager:
        raise HTTPException(status_code=503, detail="Scheduler not available")
    
    return scheduler_manager.get_status()


@app.get("/api/scheduler/tasks")
async def get_scheduled_tasks() -> List[Dict[str, Any]]:
    """Get all scheduled tasks."""
    if not scheduler_manager:
        raise HTTPException(status_code=503, detail="Scheduler not available")
    
    tasks = scheduler_manager.get_all_tasks()
    
    return [
        {
            "id": task.id,
            "name": task.name,
            "collector_type": task.collector_type,
            "schedule_pattern": task.schedule_pattern,
            "priority": task.priority.name,
            "enabled": task.enabled,
            "last_run": task.last_run.isoformat() if task.last_run else None,
            "next_run": task.next_run.isoformat() if task.next_run else None,
            "status": task.status.value,
            "error_message": task.error_message
        }
        for task in tasks
    ]


@app.post("/api/scheduler/tasks/{task_id}/run")
async def run_task_now(task_id: str) -> Dict[str, Any]:
    """Execute a task immediately."""
    if not scheduler_manager:
        raise HTTPException(status_code=503, detail="Scheduler not available")
    
    task = scheduler_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
    
    try:
        execution = await scheduler_manager.run_task_now(task_id)
        
        return {
            "task_id": execution.task_id,
            "execution_id": execution.execution_id,
            "started_at": execution.started_at.isoformat(),
            "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
            "status": execution.status.value,
            "results_count": execution.results_count,
            "duration_seconds": execution.duration_seconds,
            "error_message": execution.error_message
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to run task: {e}")


@app.put("/api/scheduler/tasks/{task_id}/enable")
async def enable_task(task_id: str) -> Dict[str, str]:
    """Enable a scheduled task."""
    if not scheduler_manager:
        raise HTTPException(status_code=503, detail="Scheduler not available")
    
    task = scheduler_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
    
    try:
        await scheduler_manager.enable_task(task_id)
        return {"message": f"Task enabled: {task.name}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to enable task: {e}")


@app.put("/api/scheduler/tasks/{task_id}/disable")
async def disable_task(task_id: str) -> Dict[str, str]:
    """Disable a scheduled task."""
    if not scheduler_manager:
        raise HTTPException(status_code=503, detail="Scheduler not available")
    
    task = scheduler_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
    
    try:
        await scheduler_manager.disable_task(task_id)
        return {"message": f"Task disabled: {task.name}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to disable task: {e}")


@app.get("/api/scheduler/history")
async def get_task_history(
    task_id: Optional[str] = Query(None, description="Filter by task ID"),
    limit: int = Query(50, description="Maximum results")
) -> List[Dict[str, Any]]:
    """Get task execution history."""
    if not scheduler_manager:
        raise HTTPException(status_code=503, detail="Scheduler not available")
    
    history = scheduler_manager.get_task_history(task_id, limit)
    
    return [
        {
            "task_id": exec.task_id,
            "execution_id": exec.execution_id,
            "started_at": exec.started_at.isoformat(),
            "completed_at": exec.completed_at.isoformat() if exec.completed_at else None,
            "status": exec.status.value,
            "results_count": exec.results_count,
            "duration_seconds": exec.duration_seconds,
            "error_message": exec.error_message
        }
        for exec in history
    ]


@app.post("/api/scheduler/presets/{preset}")
async def load_preset_config(preset: str) -> Dict[str, Any]:
    """Load a preset task configuration."""
    if not scheduler_manager:
        raise HTTPException(status_code=503, detail="Scheduler not available")
    
    valid_presets = ["default", "research", "business", "development"]
    if preset not in valid_presets:
        raise HTTPException(status_code=400, detail=f"Invalid preset. Must be one of: {valid_presets}")
    
    try:
        await scheduler_manager.load_preset_config(preset)
        tasks = scheduler_manager.get_all_tasks()
        
        return {
            "message": f"Loaded preset configuration: {preset}",
            "tasks_loaded": len(tasks),
            "tasks": [
                {
                    "id": task.id,
                    "name": task.name,
                    "schedule_pattern": task.schedule_pattern
                }
                for task in tasks
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load preset: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)