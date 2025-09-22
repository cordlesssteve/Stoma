"""Main CLI interface for KnowHunt."""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from ..collectors.arxiv import ArXivCollector
from ..collectors.github import GitHubCollector
from ..collectors.sec_edgar import SECEdgarCollector
from ..normalizers.base import AcademicNormalizer
from ..normalizers.code_projects import CodeProjectsNormalizer
from ..normalizers.public_docs import PublicDocsNormalizer
from ..storage.base import PostgreSQLStorage
from ..config.settings import load_config
from ..scheduler.manager import SchedulerManager


console = Console()


@click.group()
@click.version_option(version="0.1.0")
def main():
    """KnowHunt - Research Intelligence System"""
    pass


@main.command()
@click.option("--query", "-q", default="machine learning", help="Search query")
@click.option("--category", "-c", help="ArXiv category (e.g., cs.AI)")
@click.option("--max-results", "-n", default=10, help="Maximum number of results")
@click.option("--output", "-o", help="Output file (JSON format)")
def collect_arxiv(query: str, category: Optional[str], max_results: int, output: Optional[str]):
    """Collect papers from ArXiv."""
    
    async def _collect():
        config = {
            "max_results": max_results,
            "rate_limit": 1.0
        }
        
        collector = ArXivCollector(config)
        normalizer = AcademicNormalizer({})
        
        results = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(f"Collecting papers for: {query}", total=None)
            
            async for result in collector.collect(
                search_query=query, 
                category=category,
                start=0
            ):
                if result.success:
                    normalized = await normalizer.normalize(result)
                    results.append({
                        "id": normalized.id,
                        "title": normalized.title,
                        "authors": normalized.authors,
                        "published_date": normalized.published_date.isoformat() if normalized.published_date else None,
                        "url": normalized.url,
                        "categories": normalized.categories,
                        "summary": normalized.content[:200] + "..." if len(normalized.content) > 200 else normalized.content
                    })
                    
                    progress.update(task, description=f"Collected {len(results)} papers...")
                else:
                    console.print(f"[red]Error: {result.error_message}[/red]")
        
        return results
    
    # Run the async collection
    results = asyncio.run(_collect())
    
    if results:
        console.print(f"\n[green]Successfully collected {len(results)} papers[/green]")
        
        # Display results in a table
        table = Table(title=f"ArXiv Papers: {query}")
        table.add_column("Title", style="cyan", no_wrap=False, max_width=50)
        table.add_column("Authors", style="magenta", max_width=30)
        table.add_column("Published", style="green")
        table.add_column("Categories", style="yellow", max_width=20)
        
        for paper in results[:10]:  # Show first 10 in table
            authors_str = ", ".join(paper["authors"][:3])  # First 3 authors
            if len(paper["authors"]) > 3:
                authors_str += f" (+{len(paper['authors']) - 3} more)"
            
            published = paper["published_date"][:10] if paper["published_date"] else "Unknown"
            categories = ", ".join(paper["categories"][:2])  # First 2 categories
            
            table.add_row(
                paper["title"][:100] + "..." if len(paper["title"]) > 100 else paper["title"],
                authors_str,
                published,
                categories
            )
        
        console.print(table)
        
        # Save to file if requested
        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            console.print(f"\n[blue]Results saved to: {output_path}[/blue]")
    else:
        console.print("[yellow]No papers found[/yellow]")


@main.command()
@click.option("--config-file", "-c", help="Path to configuration file")
def health_check(config_file: Optional[str]):
    """Check health of all configured services."""
    
    async def _check_health():
        console.print("[bold]KnowHunt Health Check[/bold]\n")
        
        # Check ArXiv
        with console.status("[bold green]Checking ArXiv API..."):
            arxiv_collector = ArXivCollector({"rate_limit": 1.0})
            arxiv_healthy = await arxiv_collector.health_check()
        
        status_arxiv = "[green]✓ Online[/green]" if arxiv_healthy else "[red]✗ Offline[/red]"
        console.print(f"ArXiv API: {status_arxiv}")
        
        # Check PostgreSQL if configured
        if config_file:
            try:
                config = load_config(config_file)
                if "storage" in config and config["storage"]["type"] == "postgresql":
                    with console.status("[bold green]Checking PostgreSQL..."):
                        storage = PostgreSQLStorage(config["storage"])
                        await storage.connect()
                        pg_healthy = await storage.health_check()
                        await storage.disconnect()
                    
                    status_pg = "[green]✓ Connected[/green]" if pg_healthy else "[red]✗ Connection failed[/red]"
                    console.print(f"PostgreSQL: {status_pg}")
            except Exception as e:
                console.print(f"PostgreSQL: [red]✗ Configuration error: {e}[/red]")
        else:
            console.print("PostgreSQL: [yellow]~ Not configured[/yellow]")
        
        console.print("\n[bold]Health check complete[/bold]")
    
    asyncio.run(_check_health())


@main.command()
def list_categories():
    """List available ArXiv categories."""
    
    async def _list_categories():
        collector = ArXivCollector({})
        categories = await collector.get_categories()
        
        table = Table(title="ArXiv Categories")
        table.add_column("Code", style="cyan")
        table.add_column("Description", style="white")
        
        for code, description in categories.items():
            table.add_row(code, description)
        
        console.print(table)
    
    asyncio.run(_list_categories())


@main.command()
@click.option("--search", "-s", help="Search term")
@click.option("--source-type", "-t", help="Source type filter")
@click.option("--limit", "-l", default=20, help="Maximum results")
def search(search: str, source_type: Optional[str], limit: int):
    """Search stored data."""
    console.print(f"[yellow]Search functionality requires database setup[/yellow]")
    console.print("Configure PostgreSQL storage to enable search")


@main.command()
@click.argument("config_path", type=click.Path(exists=True))
def init_storage(config_path: str):
    """Initialize storage backend from configuration."""
    
    async def _init_storage():
        try:
            config = load_config(config_path)
            
            if "storage" not in config:
                console.print("[red]No storage configuration found[/red]")
                return
            
            storage_config = config["storage"]
            
            if storage_config["type"] == "postgresql":
                with console.status("[bold green]Initializing PostgreSQL..."):
                    storage = PostgreSQLStorage(storage_config)
                    await storage.connect()
                    console.print("[green]✓ PostgreSQL initialized successfully[/green]")
                    await storage.disconnect()
            else:
                console.print(f"[yellow]Unknown storage type: {storage_config['type']}[/yellow]")
                
        except Exception as e:
            console.print(f"[red]Error initializing storage: {e}[/red]")
    
    asyncio.run(_init_storage())


@main.command()
@click.option("--language", "-l", help="Programming language (e.g., python, javascript)")
@click.option("--period", "-p", default="week", help="Time period (week, month, year)")
@click.option("--max-results", "-n", default=10, help="Maximum number of results")
@click.option("--token", "-t", help="GitHub API token (optional)")
@click.option("--output", "-o", help="Output file (JSON format)")
def collect_github_trending(language: Optional[str], period: str, max_results: int, token: Optional[str], output: Optional[str]):
    """Collect trending repositories from GitHub."""
    
    async def _collect():
        config = {
            "per_page": max_results,
            "max_pages": 1,
            "rate_limit": 1.0
        }
        
        if token:
            config["token"] = token
        
        collector = GitHubCollector(config)
        normalizer = CodeProjectsNormalizer({})
        
        results = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(f"Collecting trending {language or 'all'} repositories...", total=None)
            
            async for result in collector.collect_trending_repos(
                language=language,
                created_since=period
            ):
                if result.success:
                    normalized = await normalizer.normalize(result)
                    results.append({
                        "id": normalized.id,
                        "title": normalized.title,
                        "authors": normalized.authors,
                        "url": normalized.url,
                        "language": result.data.get("language"),
                        "stars": result.data.get("stargazers_count", 0),
                        "forks": result.data.get("forks_count", 0),
                        "description": (result.data.get("description") or "")[:100] + "..." if len(result.data.get("description") or "") > 100 else (result.data.get("description") or ""),
                        "topics": result.data.get("topics", [])[:3]  # First 3 topics
                    })
                    
                    progress.update(task, description=f"Collected {len(results)} repositories...")
                else:
                    console.print(f"[red]Error: {result.error_message}[/red]")
        
        return results
    
    # Run the async collection
    results = asyncio.run(_collect())
    
    if results:
        console.print(f"\n[green]Successfully collected {len(results)} repositories[/green]")
        
        # Display results in a table
        table = Table(title=f"Trending GitHub Repositories ({language or 'All Languages'})")
        table.add_column("Repository", style="cyan", no_wrap=False, max_width=40)
        table.add_column("Language", style="magenta")
        table.add_column("Stars", style="yellow")
        table.add_column("Forks", style="green")
        table.add_column("Description", style="white", max_width=50)
        
        for repo in results:
            table.add_row(
                repo["title"],
                repo["language"] or "N/A",
                str(repo["stars"]),
                str(repo["forks"]),
                repo["description"]
            )
        
        console.print(table)
        
        # Save to file if requested
        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            console.print(f"\n[blue]Results saved to: {output_path}[/blue]")
    else:
        console.print("[yellow]No repositories found[/yellow]")


@main.command()
@click.argument("owner")
@click.argument("repo")
@click.option("--token", "-t", help="GitHub API token (optional)")
@click.option("--output", "-o", help="Output file (JSON format)")
def collect_github_repo(owner: str, repo: str, token: Optional[str], output: Optional[str]):
    """Collect detailed information about a specific GitHub repository."""
    
    async def _collect():
        config = {
            "rate_limit": 1.0
        }
        
        if token:
            config["token"] = token
        
        collector = GitHubCollector(config)
        normalizer = CodeProjectsNormalizer({})
        
        results = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(f"Collecting {owner}/{repo}...", total=None)
            
            async for result in collector.collect_repo_info(owner, repo):
                if result.success:
                    normalized = await normalizer.normalize(result)
                    results.append({
                        "id": normalized.id,
                        "title": normalized.title,
                        "description": result.data.get("description", ""),
                        "language": result.data.get("language"),
                        "languages": result.data.get("languages", {}),
                        "stars": result.data.get("stargazers_count", 0),
                        "forks": result.data.get("forks_count", 0),
                        "open_issues": result.data.get("open_issues_count", 0),
                        "topics": result.data.get("topics", []),
                        "license": result.data.get("license"),
                        "created_at": result.data.get("created_at"),
                        "updated_at": result.data.get("updated_at"),
                        "url": result.data.get("html_url"),
                        "clone_url": result.data.get("clone_url"),
                        "readme_preview": result.data.get("readme_content", "")[:500] + "..." if result.data.get("readme_content") and len(result.data.get("readme_content")) > 500 else result.data.get("readme_content", "")
                    })
                    
                    progress.update(task, description="Repository data collected")
                else:
                    console.print(f"[red]Error: {result.error_message}[/red]")
                    return []
        
        return results
    
    # Run the async collection
    results = asyncio.run(_collect())
    
    if results:
        repo_data = results[0]
        console.print(f"\n[green]Successfully collected repository information[/green]")
        
        # Display repository information
        console.print(f"\n[bold cyan]{repo_data['title']}[/bold cyan]")
        console.print(f"[blue]{repo_data['url']}[/blue]")
        
        if repo_data['description']:
            console.print(f"\n{repo_data['description']}")
        
        # Stats table
        stats_table = Table(title="Repository Statistics")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="white")
        
        stats_table.add_row("Stars", str(repo_data['stars']))
        stats_table.add_row("Forks", str(repo_data['forks']))
        stats_table.add_row("Open Issues", str(repo_data['open_issues']))
        stats_table.add_row("Primary Language", repo_data['language'] or "N/A")
        stats_table.add_row("License", repo_data['license'] or "N/A")
        stats_table.add_row("Created", repo_data['created_at'][:10] if repo_data['created_at'] else "N/A")
        stats_table.add_row("Last Updated", repo_data['updated_at'][:10] if repo_data['updated_at'] else "N/A")
        
        console.print(stats_table)
        
        # Languages
        if repo_data['languages']:
            console.print(f"\n[bold]Languages:[/bold]")
            total_bytes = sum(repo_data['languages'].values())
            for lang, bytes_count in sorted(repo_data['languages'].items(), key=lambda x: x[1], reverse=True):
                percentage = (bytes_count / total_bytes) * 100 if total_bytes > 0 else 0
                console.print(f"  {lang}: {percentage:.1f}%")
        
        # Topics
        if repo_data['topics']:
            console.print(f"\n[bold]Topics:[/bold] {', '.join(repo_data['topics'])}")
        
        # README preview
        if repo_data['readme_preview']:
            console.print(f"\n[bold]README Preview:[/bold]")
            console.print(repo_data['readme_preview'])
        
        # Save to file if requested
        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            console.print(f"\n[blue]Results saved to: {output_path}[/blue]")
    else:
        console.print("[yellow]Repository not found or error occurred[/yellow]")


@main.command()
@click.option("--filing-type", "-f", help="Filing type (e.g., 10-K, 8-K, 10-Q)")
@click.option("--days-back", "-d", default=1, help="Days back to search (default: 1)")
@click.option("--max-results", "-n", default=10, help="Maximum number of results")
@click.option("--output", "-o", help="Output file (JSON format)")
def collect_sec_recent(filing_type: Optional[str], days_back: int, max_results: int, output: Optional[str]):
    """Collect recent SEC filings."""
    
    async def _collect():
        config = {
            "rate_limit": 10.0  # SEC allows higher rate limits
        }
        
        collector = SECEdgarCollector(config)
        normalizer = PublicDocsNormalizer({})
        
        results = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(f"Collecting SEC filings ({filing_type or 'all types'})...", total=None)
            
            async for result in collector.collect_recent_filings(
                filing_type=filing_type,
                days_back=days_back
            ):
                if result.success:
                    normalized = await normalizer.normalize(result)
                    results.append({
                        "id": normalized.id,
                        "title": normalized.title,
                        "company": result.data.get("company_name", "Unknown"),
                        "form_type": result.data.get("form_type", ""),
                        "form_description": result.data.get("form_description", ""),
                        "filing_date": result.data.get("filing_date", ""),
                        "cik": result.data.get("cik", ""),
                        "url": result.data.get("filing_url", ""),
                        "importance": normalized.metrics.get("importance_score", 0)
                    })
                    
                    progress.update(task, description=f"Collected {len(results)} filings...")
                else:
                    console.print(f"[red]Error: {result.error_message}[/red]")
                
                if len(results) >= max_results:
                    break
        
        return results
    
    # Run the async collection
    results = asyncio.run(_collect())
    
    if results:
        console.print(f"\n[green]Successfully collected {len(results)} SEC filings[/green]")
        
        # Sort by importance score
        results.sort(key=lambda x: x["importance"], reverse=True)
        
        # Display results in a table
        table = Table(title=f"Recent SEC Filings ({filing_type or 'All Types'})")
        table.add_column("Company", style="cyan", max_width=30)
        table.add_column("Form", style="magenta")
        table.add_column("Description", style="white", max_width=25)
        table.add_column("Date", style="green")
        table.add_column("Importance", style="yellow")
        
        for filing in results:
            importance_color = "red" if filing["importance"] >= 8 else "yellow" if filing["importance"] >= 6 else "white"
            
            table.add_row(
                filing["company"][:30] + "..." if len(filing["company"]) > 30 else filing["company"],
                filing["form_type"],
                filing["form_description"][:25] + "..." if len(filing["form_description"]) > 25 else filing["form_description"],
                filing["filing_date"],
                f"[{importance_color}]{filing['importance']}[/{importance_color}]"
            )
        
        console.print(table)
        
        # Save to file if requested
        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            console.print(f"\n[blue]Results saved to: {output_path}[/blue]")
    else:
        console.print("[yellow]No SEC filings found[/yellow]")


@main.command()
@click.option("--ticker", "-t", help="Company stock ticker symbol")
@click.option("--cik", "-c", help="Company CIK number")
@click.option("--filing-type", "-f", help="Filing type filter (e.g., 10-K, 8-K)")
@click.option("--count", "-n", default=10, help="Number of filings to retrieve")
@click.option("--output", "-o", help="Output file (JSON format)")
def collect_sec_company(ticker: Optional[str], cik: Optional[str], filing_type: Optional[str], count: int, output: Optional[str]):
    """Collect SEC filings for a specific company."""
    
    if not ticker and not cik:
        console.print("[red]Error: Either --ticker or --cik must be provided[/red]")
        return
    
    async def _collect():
        config = {
            "rate_limit": 10.0
        }
        
        collector = SECEdgarCollector(config)
        normalizer = PublicDocsNormalizer({})
        
        results = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            identifier = ticker or cik
            task = progress.add_task(f"Collecting filings for {identifier}...", total=None)
            
            async for result in collector.collect_company_filings(
                company_ticker=ticker,
                cik=cik,
                filing_type=filing_type,
                count=count
            ):
                if result.success:
                    normalized = await normalizer.normalize(result)
                    results.append({
                        "id": normalized.id,
                        "title": normalized.title,
                        "company": result.data.get("company_name", "Unknown"),
                        "form_type": result.data.get("form_type", ""),
                        "form_description": result.data.get("form_description", ""),
                        "filing_date": result.data.get("filing_date", ""),
                        "summary": result.data.get("summary", "")[:200] + "..." if len(result.data.get("summary", "")) > 200 else result.data.get("summary", ""),
                        "url": result.data.get("filing_url", ""),
                        "importance": normalized.metrics.get("importance_score", 0)
                    })
                    
                    progress.update(task, description=f"Collected {len(results)} filings...")
                else:
                    console.print(f"[red]Error: {result.error_message}[/red]")
        
        return results
    
    # Run the async collection
    results = asyncio.run(_collect())
    
    if results:
        console.print(f"\n[green]Successfully collected {len(results)} filings for {ticker or cik}[/green]")
        
        # Display results
        if results:
            company_name = results[0]["company"] if results else "Unknown Company"
            
            console.print(f"\n[bold cyan]SEC Filings for {company_name}[/bold cyan]")
            console.print(f"Identifier: {ticker or cik}")
            
            table = Table(title="Company SEC Filings")
            table.add_column("Form Type", style="magenta")
            table.add_column("Description", style="white", max_width=30)
            table.add_column("Filing Date", style="green")
            table.add_column("Summary", style="cyan", max_width=50)
            
            for filing in results:
                table.add_row(
                    filing["form_type"],
                    filing["form_description"],
                    filing["filing_date"],
                    filing["summary"]
                )
            
            console.print(table)
        
        # Save to file if requested
        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            console.print(f"\n[blue]Results saved to: {output_path}[/blue]")
    else:
        console.print(f"[yellow]No filings found for {ticker or cik}[/yellow]")


@main.command()
def list_filing_types():
    """List supported SEC filing types."""
    
    collector = SECEdgarCollector({})
    filing_types = collector.get_supported_filing_types()
    
    table = Table(title="Supported SEC Filing Types")
    table.add_column("Form Type", style="cyan")
    table.add_column("Description", style="white")
    
    for form_type, description in filing_types.items():
        table.add_row(form_type, description)
    
    console.print(table)


# Scheduler Commands
@main.group()
def scheduler():
    """Manage scheduled data collection tasks."""
    pass


@scheduler.command()
@click.option("--preset", "-p", default="default", 
              type=click.Choice(["default", "research", "business", "development"]),
              help="Preset configuration to load")
def start(preset: str):
    """Start the scheduler daemon."""
    
    async def _start():
        console.print(f"[green]Starting KnowHunt scheduler with preset: {preset}[/green]")
        
        manager = SchedulerManager()
        
        try:
            await manager.initialize()
            
            if preset != "default":
                await manager.load_preset_config(preset)
            
            await manager.start()
            
            console.print("[green]✓ Scheduler started successfully[/green]")
            console.print("Press Ctrl+C to stop the scheduler")
            
            # Keep running until interrupted
            try:
                while True:
                    await asyncio.sleep(30)
                    
                    # Display status periodically
                    status = manager.get_status()
                    console.print(f"[blue]Status: {status['running_tasks']} running, "
                                f"{status['enabled_tasks']} enabled tasks[/blue]")
                    
            except KeyboardInterrupt:
                console.print("\n[yellow]Shutdown requested...[/yellow]")
        
        finally:
            await manager.stop()
            console.print("[green]✓ Scheduler stopped[/green]")
    
    asyncio.run(_start())


@scheduler.command()
def status():
    """Show scheduler status and task information."""
    
    async def _status():
        manager = SchedulerManager()
        await manager.initialize()
        
        status = manager.get_status()
        
        # Status overview
        console.print(f"\n[bold cyan]Scheduler Status: {status['status'].upper()}[/bold cyan]")
        console.print(f"Total Tasks: {status['total_tasks']}")
        console.print(f"Enabled Tasks: {status['enabled_tasks']}")
        console.print(f"Running Tasks: {status['running_tasks']}")
        
        # Next scheduled runs
        if status['next_runs']:
            console.print("\n[bold]Next Scheduled Runs:[/bold]")
            table = Table()
            table.add_column("Task Name", style="cyan")
            table.add_column("Next Run", style="green")
            table.add_column("Priority", style="yellow")
            
            for run in status['next_runs']:
                next_run_time = datetime.fromisoformat(run['next_run'])
                relative_time = next_run_time - datetime.now()
                
                if relative_time.total_seconds() > 0:
                    hours = int(relative_time.total_seconds() // 3600)
                    minutes = int((relative_time.total_seconds() % 3600) // 60)
                    time_str = f"in {hours}h {minutes}m"
                else:
                    time_str = "overdue"
                
                table.add_row(
                    run['task_name'],
                    f"{next_run_time.strftime('%Y-%m-%d %H:%M')} ({time_str})",
                    run['priority']
                )
            
            console.print(table)
        
        # Recent executions
        if status['recent_executions']:
            console.print("\n[bold]Recent Executions:[/bold]")
            table = Table()
            table.add_column("Task ID", style="cyan")
            table.add_column("Started", style="green")
            table.add_column("Status", style="yellow")
            table.add_column("Results", style="white")
            table.add_column("Duration", style="magenta")
            
            for exec in status['recent_executions'][:10]:
                duration = f"{exec['duration_seconds']:.1f}s" if exec['duration_seconds'] else "N/A"
                status_color = {
                    "completed": "green",
                    "failed": "red",
                    "running": "yellow"
                }.get(exec['status'], "white")
                
                table.add_row(
                    exec['task_id'],
                    datetime.fromisoformat(exec['started_at']).strftime('%m-%d %H:%M'),
                    f"[{status_color}]{exec['status']}[/{status_color}]",
                    str(exec['results_count']),
                    duration
                )
            
            console.print(table)
    
    asyncio.run(_status())


@scheduler.command()
def list_tasks():
    """List all scheduled tasks."""
    
    async def _list():
        manager = SchedulerManager()
        await manager.initialize()
        
        tasks = manager.get_all_tasks()
        
        if not tasks:
            console.print("[yellow]No scheduled tasks found[/yellow]")
            return
        
        table = Table(title="Scheduled Tasks")
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="white", max_width=30)
        table.add_column("Type", style="magenta")
        table.add_column("Schedule", style="green")
        table.add_column("Priority", style="yellow")
        table.add_column("Enabled", style="blue")
        table.add_column("Last Run", style="dim")
        
        for task in tasks:
            last_run = task.last_run.strftime('%m-%d %H:%M') if task.last_run else "Never"
            enabled_icon = "✓" if task.enabled else "✗"
            enabled_color = "green" if task.enabled else "red"
            
            table.add_row(
                task.id,
                task.name,
                task.collector_type,
                task.schedule_pattern,
                task.priority.name,
                f"[{enabled_color}]{enabled_icon}[/{enabled_color}]",
                last_run
            )
        
        console.print(table)
    
    asyncio.run(_list())


@scheduler.command()
@click.argument("task_id")
def run_task(task_id: str):
    """Run a specific task immediately."""
    
    async def _run():
        manager = SchedulerManager()
        await manager.initialize()
        
        task = manager.get_task(task_id)
        if not task:
            console.print(f"[red]Task not found: {task_id}[/red]")
            return
        
        console.print(f"[green]Running task: {task.name}[/green]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            progress_task = progress.add_task("Executing task...", total=None)
            
            execution = await manager.run_task_now(task_id)
            
            progress.update(progress_task, description="Task completed")
        
        if execution.status.value == "completed":
            console.print(f"[green]✓ Task completed successfully[/green]")
            console.print(f"Results collected: {execution.results_count}")
            console.print(f"Duration: {execution.duration_seconds:.1f} seconds")
        else:
            console.print(f"[red]✗ Task failed: {execution.error_message}[/red]")
    
    asyncio.run(_run())


@scheduler.command()
@click.argument("task_id")
def enable_task(task_id: str):
    """Enable a scheduled task."""
    
    async def _enable():
        manager = SchedulerManager()
        await manager.initialize()
        
        task = manager.get_task(task_id)
        if not task:
            console.print(f"[red]Task not found: {task_id}[/red]")
            return
        
        await manager.enable_task(task_id)
        console.print(f"[green]✓ Task enabled: {task.name}[/green]")
    
    asyncio.run(_enable())


@scheduler.command()
@click.argument("task_id")
def disable_task(task_id: str):
    """Disable a scheduled task."""
    
    async def _disable():
        manager = SchedulerManager()
        await manager.initialize()
        
        task = manager.get_task(task_id)
        if not task:
            console.print(f"[red]Task not found: {task_id}[/red]")
            return
        
        await manager.disable_task(task_id)
        console.print(f"[yellow]Task disabled: {task.name}[/yellow]")
    
    asyncio.run(_disable())


@scheduler.command()
@click.option("--preset", "-p", 
              type=click.Choice(["default", "research", "business", "development"]),
              required=True,
              help="Preset configuration to load")
def load_preset(preset: str):
    """Load a preset task configuration."""
    
    async def _load():
        manager = SchedulerManager()
        await manager.initialize()
        
        console.print(f"[green]Loading preset configuration: {preset}[/green]")
        
        await manager.load_preset_config(preset)
        
        tasks = manager.get_all_tasks()
        console.print(f"[green]✓ Loaded {len(tasks)} tasks from preset[/green]")
        
        # Show loaded tasks
        for task in tasks:
            console.print(f"  - {task.name} ({task.schedule_pattern})")
    
    asyncio.run(_load())


@scheduler.command()
@click.argument("task_id")
@click.argument("name")
@click.argument("collector_type", type=click.Choice(["arxiv", "github", "sec_edgar"]))
@click.argument("schedule_pattern")
@click.option("--priority", "-p", default="normal", 
              type=click.Choice(["low", "normal", "high", "critical"]),
              help="Task priority")
@click.option("--config", "-c", help="JSON configuration for the collector")
def add_task(task_id: str, name: str, collector_type: str, schedule_pattern: str, priority: str, config: Optional[str]):
    """Add a new scheduled task."""
    
    async def _add():
        manager = SchedulerManager()
        await manager.initialize()
        
        # Parse collector config
        collector_config = {}
        if config:
            try:
                collector_config = json.loads(config)
            except json.JSONDecodeError:
                console.print("[red]Invalid JSON configuration[/red]")
                return
        
        await manager.create_and_add_task(
            task_id=task_id,
            name=name,
            collector_type=collector_type,
            collector_config=collector_config,
            schedule_pattern=schedule_pattern,
            priority=priority
        )
        
        console.print(f"[green]✓ Task added: {name}[/green]")
        console.print(f"  ID: {task_id}")
        console.print(f"  Type: {collector_type}")
        console.print(f"  Schedule: {schedule_pattern}")
        console.print(f"  Priority: {priority}")
    
    asyncio.run(_add())


@scheduler.command()
@click.argument("task_id")
def remove_task(task_id: str):
    """Remove a scheduled task."""
    
    async def _remove():
        manager = SchedulerManager()
        await manager.initialize()
        
        task = manager.get_task(task_id)
        if not task:
            console.print(f"[red]Task not found: {task_id}[/red]")
            return
        
        await manager.remove_task(task_id)
        console.print(f"[yellow]Task removed: {task.name}[/yellow]")
    
    asyncio.run(_remove())


if __name__ == "__main__":
    main()