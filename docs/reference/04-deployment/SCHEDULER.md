# Stoma Scheduler System

The Stoma scheduler provides automated data collection from various sources on configurable schedules.

## Overview

The scheduler system consists of:

- **Scheduler Engine**: Executes tasks based on cron-like schedules
- **Task Management**: Create, modify, enable/disable scheduled tasks
- **Preset Configurations**: Pre-built task sets for different use cases
- **CLI Interface**: Command-line management tools
- **Web API**: REST endpoints for scheduler control
- **Systemd Service**: Run as a background daemon

## Quick Start

### 1. Basic CLI Usage

```bash
# Start the scheduler with default tasks
stoma scheduler start

# View scheduler status
stoma scheduler status

# List all scheduled tasks
stoma scheduler list-tasks

# Run a task immediately
stoma scheduler run-task arxiv_daily_trending

# Load a preset configuration
stoma scheduler load-preset --preset research
```

### 2. Web Dashboard

Start the web server:
```bash
python -m stoma.api.main
```

Visit `http://localhost:8000` and use the scheduler API endpoints:
- `GET /api/scheduler/status` - Get scheduler status
- `GET /api/scheduler/tasks` - List all tasks
- `POST /api/scheduler/tasks/{task_id}/run` - Run task immediately
- `PUT /api/scheduler/tasks/{task_id}/enable` - Enable task
- `PUT /api/scheduler/tasks/{task_id}/disable` - Disable task

### 3. As a Service

Install and run as a systemd service:
```bash
# Install the service
./scripts/install-scheduler-service.sh

# Start the service
sudo systemctl start stoma-scheduler

# Check status
sudo systemctl status stoma-scheduler

# View logs
sudo journalctl -u stoma-scheduler -f
```

## Task Configuration

### Schedule Patterns

The scheduler supports several schedule pattern formats:

- `hourly` - Every hour
- `daily` - Every day at midnight
- `weekly` - Every Sunday at midnight
- `daily_at_HH:MM` - Daily at specific time (e.g., `daily_at_09:00`)
- `interval_XXm` - Every XX minutes (e.g., `interval_30m`)
- `interval_XXh` - Every XX hours (e.g., `interval_2h`)

### Task Priorities

- `critical` - Highest priority, runs first
- `high` - High priority
- `normal` - Default priority
- `low` - Lowest priority

### Collector Types

- `arxiv` - Academic papers from ArXiv
- `github` - GitHub repositories and projects
- `sec_edgar` - SEC corporate filings

## Preset Configurations

### Default Preset
Comprehensive collection across all source types:
- ArXiv trending papers (daily)
- GitHub trending repositories (daily)
- SEC recent filings (every 2 hours)
- Weekly comprehensive scans

### Research Preset
Focused on academic research:
- ArXiv AI research papers
- ArXiv NLP research papers
- ArXiv computer vision papers

### Business Preset
Corporate intelligence focus:
- SEC 10-K annual reports
- SEC 10-Q quarterly reports
- SEC insider trading reports

### Development Preset
Software development tracking:
- GitHub Python trending
- GitHub TypeScript trending
- GitHub Rust trending
- GitHub Go trending

## Custom Task Creation

### CLI Method

```bash
stoma scheduler add-task \\
  my_custom_task \\
  "Custom ArXiv ML Scan" \\
  arxiv \\
  "daily_at_08:00" \\
  --priority high \\
  --config '{"search_query": "machine learning", "max_results": 25}'
```

### Programmatic Method

```python
from stoma.scheduler.config import create_custom_task
from stoma.scheduler.base import TaskPriority

task = create_custom_task(
    task_id="ml_papers_scan",
    name="Daily ML Papers Scan",
    collector_type="arxiv",
    collector_config={
        "search_query": "cat:cs.LG OR cat:cs.AI",
        "max_results": 50,
        "sort_by": "submittedDate"
    },
    schedule_pattern="daily_at_09:00",
    priority=TaskPriority.HIGH
)
```

## Configuration Examples

### ArXiv Collector Config
```json
{
  "search_query": "cat:cs.AI OR cat:cs.LG",
  "max_results": 50,
  "sort_by": "submittedDate",
  "sort_order": "descending"
}
```

### GitHub Collector Config
```json
{
  "language": "python",
  "since": "daily",
  "limit": 30
}
```

### SEC EDGAR Collector Config
```json
{
  "days_back": 1,
  "filing_type": "8-K"
}
```

## Monitoring and Management

### Task Execution History

View recent task executions:
```bash
stoma scheduler status
```

Monitor execution history via API:
```bash
curl http://localhost:8000/api/scheduler/history
```

### Error Handling

Tasks that fail are automatically retried based on configuration:
- Default retry count: 3
- Exponential backoff between retries
- Failed tasks are logged with error details

### Resource Management

The scheduler includes built-in resource management:
- Maximum concurrent tasks (default: 3)
- Task timeout limits (configurable per task)
- Rate limiting for API collectors
- Memory and CPU usage monitoring

## Storage Integration

Tasks automatically store collected data when storage is configured:

1. Data is collected by the specified collector
2. Results are normalized using appropriate normalizer
3. Normalized data is stored in PostgreSQL
4. Storage connection is shared across all tasks

## Advanced Usage

### Custom Normalizers

Create task-specific data processing:
```python
from stoma.normalizers.base import BaseNormalizer

class CustomNormalizer(BaseNormalizer):
    async def normalize(self, result):
        # Custom processing logic
        return normalized_data
```

### Task Dependencies

Implement task chains or dependencies:
```python
# Run secondary task after primary completes
if primary_task.status == TaskStatus.COMPLETED:
    await scheduler.run_task_now("secondary_task_id")
```

### Conditional Execution

Add conditions for task execution:
```python
# Only run during business hours
if 9 <= datetime.now().hour <= 17:
    await scheduler.run_task_now("business_hours_task")
```

## Troubleshooting

### Common Issues

1. **Task fails with import error**
   - Check collector class names in scheduler/base.py
   - Ensure all dependencies are installed

2. **No data being collected**
   - Verify storage configuration
   - Check collector-specific settings (API keys, rate limits)

3. **Tasks not running on schedule**
   - Check system time and timezone
   - Verify schedule pattern syntax
   - Ensure tasks are enabled

4. **High memory usage**
   - Reduce max_concurrent_tasks
   - Lower result limits in collector configs
   - Implement result streaming for large datasets

### Debugging

Enable verbose logging:
```bash
export PYTHONPATH=/path/to/stoma
python -m stoma.scheduler.manager --log-level DEBUG
```

Check scheduler state file:
```bash
cat scheduler_state.json
```

### Performance Tuning

1. **Optimize concurrent tasks**:
   ```python
   scheduler_config = {
       "max_concurrent_tasks": 5,  # Increase for more parallelism
       "check_interval_seconds": 30  # More frequent checks
   }
   ```

2. **Collector-specific optimizations**:
   - Use appropriate rate limits for APIs
   - Batch requests where possible
   - Implement caching for repeated queries

3. **Database optimization**:
   - Regular VACUUM and ANALYZE operations
   - Appropriate indexing on search columns
   - Connection pooling for high-throughput scenarios

## API Reference

### REST Endpoints

- `GET /api/scheduler/status` - Scheduler status and statistics
- `GET /api/scheduler/tasks` - List all scheduled tasks
- `POST /api/scheduler/tasks/{task_id}/run` - Execute task immediately
- `PUT /api/scheduler/tasks/{task_id}/enable` - Enable task
- `PUT /api/scheduler/tasks/{task_id}/disable` - Disable task
- `GET /api/scheduler/history` - Task execution history
- `POST /api/scheduler/presets/{preset}` - Load preset configuration

### CLI Commands

```bash
stoma scheduler --help
stoma scheduler start [--preset PRESET]
stoma scheduler status
stoma scheduler list-tasks
stoma scheduler run-task TASK_ID
stoma scheduler enable-task TASK_ID
stoma scheduler disable-task TASK_ID
stoma scheduler load-preset --preset PRESET
stoma scheduler add-task TASK_ID NAME TYPE SCHEDULE [OPTIONS]
stoma scheduler remove-task TASK_ID
```

## Security Considerations

- Service runs with restricted user permissions
- No new privileges escalation
- Protected system directories
- Resource limits enforced
- Audit logging enabled

## Future Enhancements

Planned features for future releases:

1. **Advanced Scheduling**:
   - Cron expression support
   - Timezone-aware scheduling
   - Holiday/weekend handling

2. **Distributed Execution**:
   - Multi-node scheduler support
   - Load balancing across workers
   - Fault tolerance and failover

3. **Enhanced Monitoring**:
   - Metrics collection and dashboards
   - Alert notifications
   - Performance analytics

4. **Workflow Management**:
   - Task dependencies and chains
   - Conditional execution
   - Data pipeline orchestration