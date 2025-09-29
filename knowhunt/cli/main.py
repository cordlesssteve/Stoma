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
from ..analysis.nlp_service import NLPService
from ..analysis.batch_processor import BatchProcessor
from ..analysis.trend_detector import TrendDetector
from ..analysis.correlation_analyzer import CorrelationAnalyzer
from ..analysis.llm_analyzer import LLMAnalyzer
from ..storage.report_manager import ReportStorageManager
from ..integrations.deep_research_bridge import (
    DeepResearchBridge,
    DeepResearchConfig,
    analyze_papers_with_deep_research,
    create_default_deep_research_config
)
from ..config.deep_research import (
    build_deep_research_config_from_settings,
    is_deep_research_enabled,
    validate_deep_research_config
)


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
                        "summary": normalized.content[:200] + "..." if len(normalized.content) > 200 else normalized.content,
                        "full_content": normalized.content  # Keep full content for LLM analysis
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


# NLP Analysis Commands
@main.group()
def nlp():
    """Natural Language Processing analysis commands."""
    pass


# LLM Analysis Commands
@main.group()
def llm():
    """Large Language Model analysis commands."""
    pass


@nlp.command()
@click.argument("paper_id", type=int)
def analyze_paper(paper_id: int):
    """Analyze a specific paper from the database."""
    
    console.print(f"[green]Analyzing paper {paper_id}...[/green]")
    
    service = NLPService()
    result = service.analyze_paper(paper_id)
    
    if result:
        console.print(f"\n[bold cyan]Analysis Results for Paper {paper_id}[/bold cyan]")
        
        # Summary
        console.print(f"\n[bold]Summary:[/bold]")
        console.print(result.summary[:500] + "..." if len(result.summary) > 500 else result.summary)
        
        # Keywords
        console.print(f"\n[bold]Top Keywords:[/bold]")
        for keyword, score in result.keywords[:5]:
            console.print(f"  • {keyword}: {score:.3f}")
        
        # Entities
        console.print(f"\n[bold]Named Entities:[/bold]")
        for entity_type, entities in list(result.entities.items())[:3]:
            if entities:
                console.print(f"  {entity_type}: {', '.join(entities[:3])}")
        
        # Sentiment
        console.print(f"\n[bold]Sentiment Analysis:[/bold]")
        console.print(f"  Polarity: {result.sentiment.get('polarity', 0):.3f}")
        console.print(f"  Subjectivity: {result.sentiment.get('subjectivity', 0):.3f}")
        console.print(f"  Label: {result.sentiment.get('sentiment_label', 'neutral')}")
        
        # Statistics
        console.print(f"\n[bold]Document Statistics:[/bold]")
        console.print(f"  Word Count: {result.word_count:,}")
        console.print(f"  Sentences: {result.sentence_count:,}")
        console.print(f"  Readability Score: {result.readability_score:.1f}/100")
        
    else:
        console.print(f"[red]Failed to analyze paper {paper_id}[/red]")


@nlp.command()
@click.argument("file_path", type=click.Path(exists=True))
@click.option("--document-id", "-d", help="Optional document identifier")
def analyze_file(file_path: str, document_id: Optional[str]):
    """Analyze a document from file."""
    
    console.print(f"[green]Analyzing file: {file_path}[/green]")
    
    service = NLPService()
    result = service.analyze_document(file_path, document_id)
    
    if result:
        _display_analysis_results(result)
    else:
        console.print(f"[red]Failed to analyze file: {file_path}[/red]")


@nlp.command()
@click.argument("text")
@click.option("--save/--no-save", default=True, help="Save analysis to database")
def analyze_text(text: str, save: bool):
    """Analyze raw text."""
    
    console.print("[green]Analyzing text...[/green]")
    
    service = NLPService()
    result = service.analyze_text(text, store_result=save)
    
    if result:
        _display_analysis_results(result)
    else:
        console.print("[red]Failed to analyze text[/red]")


@nlp.command()
@click.option("--limit", "-l", default=100, help="Maximum papers to process")
@click.option("--paper-ids", "-p", multiple=True, type=int, help="Specific paper IDs to analyze")
def batch_analyze(limit: int, paper_ids: tuple):
    """Analyze multiple papers in batch."""
    
    paper_list = list(paper_ids) if paper_ids else None
    
    console.print(f"[green]Starting batch analysis...[/green]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Processing papers...", total=None)
        
        service = NLPService()
        summary = service.batch_analyze_papers(paper_list, limit)
        
        progress.update(task, description="Batch analysis complete")
    
    # Display results
    console.print(f"\n[bold cyan]Batch Analysis Summary[/bold cyan]")
    console.print(f"Total Processed: {summary['total_processed']}")
    console.print(f"Successful: [green]{summary['successful']}[/green]")
    console.print(f"Failed: [red]{summary['failed']}[/red]")
    console.print(f"Skipped: [yellow]{summary['skipped']}[/yellow]")
    console.print(f"Processing Time: {summary['processing_time_ms']/1000:.1f} seconds")
    
    if summary.get('error'):
        console.print(f"\n[red]Error: {summary['error']}[/red]")


@nlp.command()
def stats():
    """Show NLP analysis statistics."""
    
    service = NLPService()
    
    # Analysis summary
    summary = service.get_analysis_summary()
    
    if summary:
        console.print("\n[bold cyan]NLP Analysis Statistics[/bold cyan]")
        
        table = Table()
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        
        table.add_row("Total Analyses", str(summary.get('total_analyses', 0)))
        table.add_row("Analyzed Papers", str(summary.get('analyzed_papers', 0)))
        table.add_row("Avg Word Count", f"{summary.get('avg_word_count', 0):.0f}")
        table.add_row("Avg Readability", f"{summary.get('avg_readability', 0):.1f}/100")
        table.add_row("Avg Processing Time", f"{summary.get('avg_processing_time_ms', 0):.0f}ms")
        
        if summary.get('first_analysis'):
            table.add_row("First Analysis", summary['first_analysis'][:10])
        if summary.get('last_analysis'):
            table.add_row("Last Analysis", summary['last_analysis'][:10])
        
        console.print(table)
    
    # Top keywords
    top_keywords = service.nlp_storage.get_top_keywords(10)
    
    if top_keywords:
        console.print("\n[bold]Top Keywords Across All Documents:[/bold]")
        
        table = Table()
        table.add_column("Keyword", style="cyan")
        table.add_column("Documents", style="yellow")
        table.add_column("Avg Score", style="green")
        
        for kw in top_keywords:
            table.add_row(
                kw['keyword'],
                str(kw['document_count']),
                f"{kw['avg_score']:.3f}"
            )
        
        console.print(table)
    
    # Top entities
    top_entities = service.nlp_storage.get_top_entities(limit=10)
    
    if top_entities:
        console.print("\n[bold]Top Entities Across All Documents:[/bold]")
        
        table = Table()
        table.add_column("Entity", style="cyan")
        table.add_column("Type", style="magenta")
        table.add_column("Documents", style="yellow")
        
        for entity in top_entities:
            table.add_row(
                entity['entity_text'][:30],
                entity['entity_type'],
                str(entity['document_count'])
            )
        
        console.print(table)
    
    # Sentiment distribution
    sentiment_dist = service.nlp_storage.get_sentiment_distribution()
    
    if sentiment_dist and sentiment_dist.get('total'):
        console.print("\n[bold]Sentiment Distribution:[/bold]")
        
        total = sentiment_dist['total']
        positive = sentiment_dist.get('positive_count', 0)
        negative = sentiment_dist.get('negative_count', 0)
        neutral = sentiment_dist.get('neutral_count', 0)
        
        console.print(f"  Positive: [green]{positive} ({positive/total*100:.1f}%)[/green]")
        console.print(f"  Negative: [red]{negative} ({negative/total*100:.1f}%)[/red]")
        console.print(f"  Neutral: [yellow]{neutral} ({neutral/total*100:.1f}%)[/yellow]")
        console.print(f"  Avg Polarity: {sentiment_dist.get('avg_polarity', 0):.3f}")
        console.print(f"  Avg Subjectivity: {sentiment_dist.get('avg_subjectivity', 0):.3f}")


@nlp.command()
@click.option("--keywords", "-k", multiple=True, help="Keywords to search for")
@click.option("--entity-type", "-e", help="Entity type to search")
@click.option("--entity-text", "-t", help="Entity text to search")
@click.option("--limit", "-l", default=10, help="Maximum results")
def search_analysis(keywords: tuple, entity_type: str, entity_text: str, limit: int):
    """Search analyzed documents."""
    
    service = NLPService()
    
    if keywords:
        console.print(f"[green]Searching by keywords: {', '.join(keywords)}[/green]")
        results = service.nlp_storage.search_by_keywords(list(keywords), limit)
        
        if results:
            table = Table(title="Search Results by Keywords")
            table.add_column("Document ID", style="cyan")
            table.add_column("Paper Title", style="white", max_width=40)
            table.add_column("Matching Keywords", style="yellow")
            table.add_column("Avg Score", style="green")
            
            for result in results:
                table.add_row(
                    result['document_id'],
                    (result.get('paper_title') or 'N/A')[:40],
                    str(result.get('matching_keywords', 0)),
                    f"{result.get('avg_keyword_score', 0):.3f}"
                )
            
            console.print(table)
        else:
            console.print("[yellow]No matching documents found[/yellow]")
    
    elif entity_type:
        console.print(f"[green]Searching by entity type: {entity_type}[/green]")
        results = service.nlp_storage.search_by_entities(entity_type, entity_text, limit)
        
        if results:
            table = Table(title="Search Results by Entity")
            table.add_column("Document ID", style="cyan")
            table.add_column("Paper Title", style="white", max_width=40)
            table.add_column("Entities", style="magenta", max_width=50)
            
            for result in results:
                entities_str = ', '.join(result.get('entities', [])[:3]) if 'entities' in result else result.get('entity_text', '')
                table.add_row(
                    result['document_id'],
                    (result.get('paper_title') or 'N/A')[:40],
                    entities_str[:50]
                )
            
            console.print(table)
        else:
            console.print("[yellow]No matching documents found[/yellow]")
    
    else:
        console.print("[yellow]Please specify keywords (-k) or entity type (-e) to search[/yellow]")


def _display_analysis_results(result):
    """Helper to display analysis results consistently."""
    console.print(f"\n[bold cyan]Analysis Results[/bold cyan]")
    
    # Summary
    console.print(f"\n[bold]Summary:[/bold]")
    console.print(result.summary[:500] + "..." if len(result.summary) > 500 else result.summary)
    
    # Keywords
    if result.keywords:
        console.print(f"\n[bold]Top Keywords:[/bold]")
        for keyword, score in result.keywords[:5]:
            console.print(f"  • {keyword}: {score:.3f}")
    
    # Entities
    if result.entities:
        console.print(f"\n[bold]Named Entities:[/bold]")
        for entity_type, entities in list(result.entities.items())[:3]:
            if entities:
                console.print(f"  {entity_type}: {', '.join(entities[:3])}")
    
    # Sentiment
    if result.sentiment:
        console.print(f"\n[bold]Sentiment Analysis:[/bold]")
        console.print(f"  Polarity: {result.sentiment.get('polarity', 0):.3f}")
        console.print(f"  Subjectivity: {result.sentiment.get('subjectivity', 0):.3f}")
        console.print(f"  Label: {result.sentiment.get('sentiment_label', 'neutral')}")
    
    # Topics
    if result.topics:
        console.print(f"\n[bold]Main Topics:[/bold]")
        for topic in result.topics[:3]:
            console.print(f"  • {topic}")
    
    # Statistics
    console.print(f"\n[bold]Document Statistics:[/bold]")
    console.print(f"  Word Count: {result.word_count:,}")
    console.print(f"  Sentences: {result.sentence_count:,}")
    console.print(f"  Readability Score: {result.readability_score:.1f}/100")


@nlp.command()
@click.option("--timeframe", "-t", default=30, help="Days to analyze for trends")
@click.option("--min-frequency", "-f", default=3, help="Minimum keyword frequency")
def detect_trends(timeframe: int, min_frequency: int):
    """Detect trending keywords and topics."""
    
    console.print(f"[green]Detecting trends over {timeframe} days...[/green]")
    
    detector = TrendDetector()
    trends = detector.detect_keyword_trends(timeframe, min_frequency)
    
    if trends:
        console.print(f"\n[bold cyan]Found {len(trends)} Trending Keywords[/bold cyan]")
        
        table = Table()
        table.add_column("Keyword", style="cyan")
        table.add_column("Trend Type", style="magenta")
        table.add_column("Strength", style="yellow")
        table.add_column("Velocity", style="green")
        table.add_column("Domains", style="blue")
        
        for trend in trends[:10]:
            domains_str = ", ".join(trend.domains[:2])
            table.add_row(
                trend.keyword,
                trend.trend_type,
                f"{trend.strength:.3f}",
                f"{trend.velocity:.3f}",
                domains_str
            )
        
        console.print(table)
        
        # Show emerging topics
        emerging = detector.detect_emerging_topics(timeframe * 3, 0.5)
        if emerging:
            console.print(f"\n[bold]Emerging Topics:[/bold]")
            for topic in emerging[:5]:
                console.print(f"  • {topic['topic']}: {topic['emergence_score']:.3f}")
    else:
        console.print("[yellow]No significant trends detected[/yellow]")


@nlp.command()
@click.option("--paper-ids", "-p", multiple=True, type=int, help="Specific paper IDs")
@click.option("--threshold", "-t", default=0.3, help="Correlation threshold")
@click.option("--max-results", "-m", default=20, help="Maximum correlations to show")
def find_correlations(paper_ids: tuple, threshold: float, max_results: int):
    """Find correlations between papers."""
    
    paper_list = list(paper_ids) if paper_ids else None
    
    console.print("[green]Finding paper correlations...[/green]")
    
    analyzer = CorrelationAnalyzer()
    correlations = analyzer.find_paper_correlations(paper_list, threshold, max_results)
    
    if correlations:
        console.print(f"\n[bold cyan]Found {len(correlations)} Paper Correlations[/bold cyan]")
        
        table = Table()
        table.add_column("Paper 1", style="cyan")
        table.add_column("Paper 2", style="cyan")
        table.add_column("Score", style="yellow")
        table.add_column("Type", style="magenta")
        table.add_column("Shared Elements", style="green")
        
        for corr in correlations:
            shared_str = ", ".join(corr.shared_elements[:3])
            table.add_row(
                str(corr.paper1_id),
                str(corr.paper2_id),
                f"{corr.correlation_score:.3f}",
                corr.correlation_type,
                shared_str
            )
        
        console.print(table)
        
        # Show topic clusters
        clusters = analyzer.cluster_papers_by_topic(90, 3)
        if clusters:
            console.print(f"\n[bold]Topic Clusters Found:[/bold]")
            for cluster in clusters[:3]:
                console.print(f"  • {cluster.primary_topic}: {len(cluster.paper_ids)} papers (coherence: {cluster.coherence_score:.3f})")
    else:
        console.print("[yellow]No significant correlations found[/yellow]")


# Batch Processing Commands
@main.group()
def batch():
    """Batch processing commands for overnight analysis."""
    pass


@batch.command()
@click.option("--paper-ids", "-p", multiple=True, type=int, help="Specific paper IDs")
@click.option("--max-papers", "-m", default=100, help="Maximum papers to process")
@click.option("--priority", default=5, help="Task priority (1-10)")
def schedule_nlp(paper_ids: tuple, max_papers: int, priority: int):
    """Schedule batch NLP analysis."""
    
    paper_list = list(paper_ids) if paper_ids else None
    
    processor = BatchProcessor()
    task_id = processor.schedule_nlp_analysis_batch(paper_list, priority, max_papers)
    
    console.print(f"[green]✓ Scheduled NLP analysis batch: {task_id}[/green]")
    console.print(f"Papers to process: {max_papers if not paper_list else len(paper_list)}")
    console.print(f"Priority: {priority}")


@batch.command()
@click.option("--timeframe", "-t", default=30, help="Days to analyze")
@click.option("--priority", default=6, help="Task priority (1-10)")
def schedule_trends(timeframe: int, priority: int):
    """Schedule batch trend analysis."""
    
    processor = BatchProcessor()
    task_id = processor.schedule_trend_analysis_batch(timeframe, priority)
    
    console.print(f"[green]✓ Scheduled trend analysis batch: {task_id}[/green]")
    console.print(f"Timeframe: {timeframe} days")
    console.print(f"Priority: {priority}")


@batch.command()
@click.option("--paper-ids", "-p", multiple=True, type=int, help="Specific paper IDs")
@click.option("--priority", default=4, help="Task priority (1-10)")
def schedule_correlations(paper_ids: tuple, priority: int):
    """Schedule batch correlation analysis."""
    
    paper_list = list(paper_ids) if paper_ids else None
    
    processor = BatchProcessor()
    task_id = processor.schedule_correlation_analysis_batch(paper_list, priority)
    
    console.print(f"[green]✓ Scheduled correlation analysis batch: {task_id}[/green]")
    console.print(f"Papers to analyze: {'All recent' if not paper_list else len(paper_list)}")
    console.print(f"Priority: {priority}")


@batch.command()
@click.option("--job-id", "-j", help="Optional job identifier")
@click.option("--max-hours", "-h", default=6, help="Maximum runtime hours")
def run_job(job_id: str, max_hours: int):
    """Run batch processing job."""
    
    async def _run():
        processor = BatchProcessor()
        
        console.print(f"[green]Starting batch job (max {max_hours}h)...[/green]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Running batch job...", total=None)
            
            result = await processor.run_batch_job(job_id, max_hours)
            
            progress.update(task, description="Batch job completed")
        
        # Display results
        console.print(f"\n[bold cyan]Batch Job Results[/bold cyan]")
        console.print(f"Job ID: {result.job_id}")
        console.print(f"Duration: {result.total_processing_time:.1f} seconds")
        console.print(f"Tasks completed: [green]{result.completed_tasks}[/green]")
        console.print(f"Tasks failed: [red]{result.failed_tasks}[/red]")
        console.print(f"Tasks skipped: [yellow]{result.skipped_tasks}[/yellow]")
        
        # Show summary results
        summary = result.results_summary
        
        if summary.get('nlp_analysis', {}).get('total_papers_processed', 0) > 0:
            nlp = summary['nlp_analysis']
            console.print(f"\n[bold]NLP Analysis:[/bold]")
            console.print(f"  Papers processed: {nlp['total_papers_processed']}")
            console.print(f"  Successful: {nlp['successful_analyses']}")
        
        if summary.get('trend_detection', {}).get('keyword_trends_found', 0) > 0:
            trends = summary['trend_detection']
            console.print(f"\n[bold]Trend Detection:[/bold]")
            console.print(f"  Keyword trends: {trends['keyword_trends_found']}")
            console.print(f"  Emerging topics: {trends['emerging_topics_found']}")
            if trends['top_keywords']:
                console.print(f"  Top keywords: {', '.join(trends['top_keywords'][:5])}")
        
        if summary.get('correlation_analysis', {}).get('correlations_found', 0) > 0:
            corr = summary['correlation_analysis']
            console.print(f"\n[bold]Correlation Analysis:[/bold]")
            console.print(f"  Correlations found: {corr['correlations_found']}")
            console.print(f"  Clusters found: {corr['clusters_found']}")
    
    import asyncio
    asyncio.run(_run())


@batch.command()
def status():
    """Show batch processing queue status."""
    
    processor = BatchProcessor()
    status = processor.get_queue_status()
    
    console.print("[bold cyan]Batch Processing Queue Status[/bold cyan]")
    
    table = Table()
    table.add_column("Metric", style="cyan")
    table.add_column("Count", style="white")
    
    table.add_row("Total Tasks", str(status['total_tasks']))
    table.add_row("Pending", str(status['pending_tasks']))
    table.add_row("Running", str(status['running_tasks']))
    table.add_row("Completed", str(status['completed_tasks']))
    table.add_row("Failed", str(status['failed_tasks']))
    
    console.print(table)
    
    console.print(f"\n[bold]Next Batch Window:[/bold] {status['next_batch_window'][:19]}")


@llm.command()
@click.argument("text")
@click.option("--provider", "-p", default="ollama", type=click.Choice(["openai", "anthropic", "ollama"]), help="LLM provider")
@click.option("--model", "-m", default="gemma2:2b", help="Model name")
@click.option("--max-tokens", default=1500, help="Maximum tokens in response")
@click.option("--temperature", "-t", default=0.1, help="Temperature for generation")
@click.option("--output", "-o", help="Output file to save the analysis report (JSON format)")
def analyze_text(text: str, provider: str, model: str, max_tokens: int, temperature: float, output: Optional[str]):
    """Analyze text using LLM for research intelligence."""
    
    async def _analyze():
        console.print(f"[green]Analyzing text with {provider} ({model})...[/green]")
        
        try:
            analyzer = LLMAnalyzer(
                provider=provider,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Running LLM analysis...", total=None)
                
                result = await analyzer.analyze_research_paper(
                    text=text,
                    title="CLI Text Analysis",
                    document_id=f"cli_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
                
                progress.update(task, description="Analysis complete")
            
            # Display results
            console.print(f"\n[bold cyan]LLM Analysis Results[/bold cyan]")
            
            console.print(f"\n[bold]Research Quality Score:[/bold] {result.research_quality_score:.2f}/10")
            
            if result.novel_contributions:
                console.print(f"\n[bold]Novel Contributions:[/bold]")
                for i, contribution in enumerate(result.novel_contributions[:3], 1):
                    console.print(f"  {i}. {contribution}")
            
            if result.technical_innovations:
                console.print(f"\n[bold]Technical Innovations:[/bold]")
                for innovation in result.technical_innovations[:3]:
                    console.print(f"  • {innovation}")
            
            if result.business_implications:
                console.print(f"\n[bold]Business Implications:[/bold]")
                for implication in result.business_implications[:3]:
                    console.print(f"  • {implication}")
            
            # Usage statistics
            stats = analyzer.get_usage_statistics()
            console.print(f"\n[bold]Analysis Statistics:[/bold]")
            console.print(f"  Tokens Used: {stats['total_tokens']}")
            console.print(f"  Success Rate: {stats['success_rate']:.1%}")
            console.print(f"  Provider: {provider}")
            
            # Save report with automatic organization or custom path
            import json
            from pathlib import Path

            report_data = {
                "timestamp": datetime.now().isoformat(),
                "input_text": text,
                "provider": provider,
                "model": model,
                "analysis": {
                    "research_quality_score": result.research_quality_score,
                    "novel_contributions": result.novel_contributions,
                    "technical_innovations": result.technical_innovations,
                    "business_implications": result.business_implications,
                    "research_significance": result.research_significance,
                    "methodology_assessment": result.methodology_assessment,
                    "impact_prediction": result.impact_prediction,
                    "research_gaps_identified": result.research_gaps_identified,
                    "related_work_connections": result.related_work_connections,
                    "concept_keywords": result.concept_keywords,
                    "document_id": result.document_id,
                    "metadata": result.metadata
                },
                "usage_statistics": stats
            }

            # Use ReportStorageManager for organized storage
            report_manager = ReportStorageManager()

            if output:
                # Save to custom path but also index in system
                output_path = Path(output)
                output_path.parent.mkdir(parents=True, exist_ok=True)

                with open(output_path, 'w') as f:
                    json.dump(report_data, f, indent=2, default=str)

                # Also index in organized system
                organized_path = report_manager.save_analysis_report(report_data, auto_path=True)

                console.print(f"\n[blue]📄 Analysis report saved to: {output_path}[/blue]")
                console.print(f"[dim]📁 Also indexed in organized storage: {organized_path.name}[/dim]")
            else:
                # Auto-generate organized path
                organized_path = report_manager.save_analysis_report(report_data, auto_path=True)
                console.print(f"\n[blue]📄 Analysis report saved to: {organized_path}[/blue]")
            
        except Exception as e:
            console.print(f"[red]Analysis failed: {e}[/red]")
            if "not available" in str(e).lower():
                console.print(f"\n[yellow]💡 To use {provider}:[/yellow]")
                if provider == "ollama":
                    console.print(f"   1. Make sure Ollama is running: ollama serve")
                    console.print(f"   2. Pull the model: ollama pull {model}")
                elif provider == "openai":
                    console.print(f"   1. Set API key: export OPENAI_API_KEY='your-key'")
                elif provider == "anthropic":
                    console.print(f"   1. Set API key: export ANTHROPIC_API_KEY='your-key'")
    
    asyncio.run(_analyze())


@llm.command()
@click.option("--query", "-q", default="machine learning", help="Search query")
@click.option("--max-results", "-n", default=3, help="Maximum number of papers")
@click.option("--provider", "-p", default="ollama", type=click.Choice(["openai", "anthropic", "ollama"]), help="LLM provider")
@click.option("--model", "-m", default="gemma2:2b", help="Model name")
def collect_and_analyze_arxiv(query: str, max_results: int, provider: str, model: str):
    """Collect ArXiv papers and immediately analyze with LLM."""
    
    async def _collect_and_analyze():
        console.print(f"[green]Collecting {max_results} papers for: {query}[/green]")
        
        # Collection phase
        config = {"max_results": max_results, "rate_limit": 1.0}
        collector = ArXivCollector(config)
        normalizer = AcademicNormalizer({})
        results = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            collect_task = progress.add_task("Collecting papers...", total=None)
            
            async for result in collector.collect(search_query=query, start=0):
                if result.success and len(results) < max_results:
                    normalized = await normalizer.normalize(result)
                    results.append({
                        "title": normalized.title,
                        "content": normalized.content,
                        "id": normalized.id
                    })
                    
                    progress.update(collect_task, description=f"Collected {len(results)} papers...")
                
                if len(results) >= max_results:
                    break
        
        if not results:
            console.print("[yellow]No papers collected[/yellow]")
            return
        
        console.print(f"[green]Successfully collected {len(results)} papers[/green]")
        
        # Analysis phase
        try:
            analyzer = LLMAnalyzer(provider=provider, model=model, max_tokens=1500, temperature=0.1)
            console.print(f"[blue]Analyzing with {provider} ({model})...[/blue]")
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                analyze_task = progress.add_task("Running LLM analysis...", total=len(results))
                
                analyzed_papers = []
                for i, paper in enumerate(results):
                    try:
                        result = await analyzer.analyze_research_paper(
                            text=paper["content"],
                            title=paper["title"],
                            document_id=paper["id"]
                        )
                        analyzed_papers.append((paper, result))
                        progress.advance(analyze_task)
                        
                    except Exception as e:
                        console.print(f"[red]Failed to analyze paper {i+1}: {e}[/red]")
                        progress.advance(analyze_task)
            
            # Display results
            console.print(f"\n[bold cyan]LLM Analysis Results for {len(analyzed_papers)} Papers[/bold cyan]")
            
            for i, (paper, analysis) in enumerate(analyzed_papers, 1):
                console.print(f"\n[bold]Paper {i}: {paper['title'][:60]}...[/bold]")
                console.print(f"Research Quality: {analysis.research_quality_score:.1f}/10")
                
                if analysis.novel_contributions:
                    console.print(f"Novel Contributions: {len(analysis.novel_contributions)}")
                    console.print(f"  • {analysis.novel_contributions[0][:100]}...")
                
                if analysis.business_implications:
                    console.print(f"Business Value: {analysis.business_implications[0][:80]}...")
            
            # Summary statistics
            stats = analyzer.get_usage_statistics()
            console.print(f"\n[bold]Analysis Summary:[/bold]")
            console.print(f"Papers Analyzed: {len(analyzed_papers)}")
            console.print(f"Total Tokens: {stats['total_tokens']}")
            console.print(f"Provider: {provider}")
            
        except Exception as e:
            console.print(f"[red]LLM analysis failed: {e}[/red]")
    
    asyncio.run(_collect_and_analyze())


@llm.command()
def test_providers():
    """Test availability of all LLM providers."""
    
    async def _test():
        console.print("[bold cyan]Testing LLM Provider Availability[/bold cyan]\n")
        
        # Test text
        test_text = "Machine learning models improve pattern recognition through training on large datasets."
        
        providers = [
            ("openai", "gpt-3.5-turbo"),
            ("anthropic", "claude-3-haiku-20240307"),
            ("ollama", "gemma2:2b")
        ]
        
        for provider, model in providers:
            console.print(f"[yellow]Testing {provider} with {model}...[/yellow]")
            
            try:
                analyzer = LLMAnalyzer(
                    provider=provider,
                    model=model,
                    max_tokens=100,
                    temperature=0.1
                )
                
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console
                ) as progress:
                    task = progress.add_task(f"Testing {provider}...", total=None)
                    
                    result = await analyzer.analyze_research_paper(
                        text=test_text,
                        title="Test Paper",
                        document_id="test_provider"
                    )
                    
                    progress.update(task, description=f"{provider} test complete")
                
                console.print(f"[green]✓ {provider} ({model}): Working[/green]")
                console.print(f"   Quality Score: {result.research_quality_score:.1f}")
                
            except Exception as e:
                console.print(f"[red]✗ {provider} ({model}): {str(e)[:60]}...[/red]")
                
                # Provide helpful tips
                if provider == "ollama" and ("not available" in str(e) or "connection" in str(e).lower()):
                    console.print(f"   [dim]💡 Try: ollama serve && ollama pull {model}[/dim]")
                elif provider == "openai" and "api_key" in str(e).lower():
                    console.print(f"   [dim]💡 Try: export OPENAI_API_KEY='your-key'[/dim]")
                elif provider == "anthropic" and "api_key" in str(e).lower():
                    console.print(f"   [dim]💡 Try: export ANTHROPIC_API_KEY='your-key'[/dim]")
            
            console.print("")
    
    asyncio.run(_test())


@llm.command()
@click.option("--query", "-q", help="Search query for document IDs or keywords")
@click.option("--provider", "-p", help="Filter by provider (openai, anthropic, ollama)")
@click.option("--min-quality", "-m", type=float, help="Minimum quality score (0-10)")
@click.option("--limit", "-l", default=20, help="Maximum number of results")
def search_reports(query: Optional[str], provider: Optional[str], min_quality: Optional[float], limit: int):
    """Search saved LLM analysis reports."""

    console.print("[green]Searching analysis reports...[/green]")

    try:
        report_manager = ReportStorageManager()
        results = report_manager.search_reports(
            query=query,
            provider=provider,
            min_quality_score=min_quality,
            limit=limit
        )

        if results:
            console.print(f"\n[bold cyan]Found {len(results)} reports[/bold cyan]")

            table = Table(title="LLM Analysis Reports")
            table.add_column("Document ID", style="cyan", max_width=30)
            table.add_column("Provider", style="magenta")
            table.add_column("Model", style="yellow", max_width=15)
            table.add_column("Quality", style="green")
            table.add_column("Contributions", style="blue")
            table.add_column("Date", style="white")

            for report in results:
                quality_color = "red" if report["research_quality_score"] < 3 else "yellow" if report["research_quality_score"] < 7 else "green"

                table.add_row(
                    report["document_id"][:30],
                    report["provider"],
                    report["model"][:15] if report["model"] else "N/A",
                    f"[{quality_color}]{report['research_quality_score']:.1f}[/{quality_color}]",
                    str(report["novel_contributions_count"]),
                    report["timestamp"][:10]
                )

            console.print(table)

            # Show some statistics
            avg_quality = sum(r["research_quality_score"] for r in results) / len(results)
            total_contributions = sum(r["novel_contributions_count"] for r in results)

            console.print(f"\n[bold]Summary:[/bold]")
            console.print(f"  Average Quality Score: {avg_quality:.2f}/10")
            console.print(f"  Total Novel Contributions: {total_contributions}")

        else:
            console.print("[yellow]No reports found matching the criteria[/yellow]")

    except Exception as e:
        console.print(f"[red]Search failed: {e}[/red]")


@llm.command()
@click.argument("document_id")
def view_report(document_id: str):
    """View detailed information about a specific analysis report."""

    try:
        report_manager = ReportStorageManager()
        report = report_manager.get_report_by_id(document_id)

        if report:
            console.print(f"\n[bold cyan]Analysis Report: {document_id}[/bold cyan]")

            # Basic info
            console.print(f"[bold]Provider:[/bold] {report.get('provider', 'N/A')}")
            console.print(f"[bold]Model:[/bold] {report.get('model', 'N/A')}")
            console.print(f"[bold]Timestamp:[/bold] {report.get('timestamp', 'N/A')[:19]}")

            analysis = report.get("analysis", {})
            console.print(f"[bold]Quality Score:[/bold] {analysis.get('research_quality_score', 0):.2f}/10")

            # Novel contributions
            contributions = analysis.get("novel_contributions", [])
            if contributions:
                console.print(f"\n[bold]Novel Contributions ({len(contributions)}):[/bold]")
                for i, contribution in enumerate(contributions[:3], 1):
                    console.print(f"  {i}. {contribution[:100]}..." if len(contribution) > 100 else f"  {i}. {contribution}")
                if len(contributions) > 3:
                    console.print(f"  ... and {len(contributions) - 3} more")

            # Technical innovations
            innovations = analysis.get("technical_innovations", [])
            if innovations:
                console.print(f"\n[bold]Technical Innovations ({len(innovations)}):[/bold]")
                for innovation in innovations[:3]:
                    if isinstance(innovation, dict):
                        console.print(f"  • {innovation.get('description', str(innovation))[:80]}...")
                    else:
                        console.print(f"  • {str(innovation)[:80]}...")

            # Business implications
            implications = analysis.get("business_implications", [])
            if implications:
                console.print(f"\n[bold]Business Implications ({len(implications)}):[/bold]")
                for implication in implications[:3]:
                    console.print(f"  • {implication[:80]}..." if len(implication) > 80 else f"  • {implication}")

            # Keywords
            keywords = analysis.get("concept_keywords", [])
            if keywords:
                console.print(f"\n[bold]Key Concepts:[/bold] {', '.join(keywords[:10])}")
                if len(keywords) > 10:
                    console.print(f"... and {len(keywords) - 10} more")

            # Usage stats
            stats = report.get("usage_statistics", {})
            if stats:
                console.print(f"\n[bold]Usage Statistics:[/bold]")
                console.print(f"  Tokens Used: {stats.get('total_tokens', 'N/A')}")
                console.print(f"  Success Rate: {stats.get('success_rate', 'N/A'):.1%}" if isinstance(stats.get('success_rate'), (int, float)) else f"  Success Rate: {stats.get('success_rate', 'N/A')}")

            # File location
            index_metadata = report.get("_index_metadata", {})
            if index_metadata:
                console.print(f"\n[dim]File: {index_metadata.get('file_path', 'N/A')}[/dim]")

        else:
            console.print(f"[yellow]Report not found: {document_id}[/yellow]")

    except Exception as e:
        console.print(f"[red]Failed to view report: {e}[/red]")


@llm.command()
def storage_stats():
    """Show statistics about stored analysis reports."""

    try:
        report_manager = ReportStorageManager()
        stats = report_manager.get_storage_statistics()

        console.print(f"\n[bold cyan]Analysis Report Storage Statistics[/bold cyan]")

        # Basic stats
        console.print(f"[bold]Total Reports:[/bold] {stats['total_reports']}")
        console.print(f"[bold]Recent Reports (30 days):[/bold] {stats['recent_activity']}")
        console.print(f"[bold]Disk Usage:[/bold] {stats['disk_usage_mb']} MB")

        # Quality distribution
        quality_dist = stats["quality_distribution"]
        console.print(f"\n[bold]Quality Distribution:[/bold]")
        console.print(f"  High Quality (≥8): [green]{quality_dist['high_quality']}[/green]")
        console.print(f"  Medium Quality (5-8): [yellow]{quality_dist['medium_quality']}[/yellow]")
        console.print(f"  Low Quality (<5): [red]{quality_dist['low_quality']}[/red]")
        console.print(f"  Average Score: {quality_dist['avg_quality_score']:.2f}/10")

        # By provider
        if stats["by_provider"]:
            console.print(f"\n[bold]By Provider:[/bold]")

            table = Table()
            table.add_column("Provider", style="cyan")
            table.add_column("Reports", style="yellow")
            table.add_column("Avg Quality", style="green")

            for provider_stat in stats["by_provider"]:
                table.add_row(
                    provider_stat["provider"],
                    str(provider_stat["count"]),
                    f"{provider_stat['avg_score']:.2f}" if provider_stat["avg_score"] else "N/A"
                )

            console.print(table)

    except Exception as e:
        console.print(f"[red]Failed to get storage statistics: {e}[/red]")


# Deep Research Commands Group
@main.group()
def deep_research():
    """Advanced deep research analysis using OpenDeepResearch integration."""
    pass


@deep_research.command()
@click.option("--query", "-q", default="machine learning", help="Search query")
@click.option("--category", "-c", help="ArXiv category (e.g., cs.AI)")
@click.option("--max-results", "-n", default=5, help="Maximum number of papers")
@click.option("--research-question", "-r", help="Custom research question")
@click.option("--model", "-m", default="openai:gpt-4.1", help="Research model to use")
@click.option("--output", "-o", help="Output directory for reports")
def analyze_papers(query: str, category: Optional[str], max_results: int,
                  research_question: Optional[str], model: str, output: Optional[str]):
    """Collect ArXiv papers and perform deep research analysis."""

    async def _analyze():
        # Load configuration
        config = load_config()

        if not is_deep_research_enabled(config):
            console.print("[red]Deep research is disabled in configuration[/red]")
            console.print("[yellow]Set ENABLE_DEEP_RESEARCH=true to enable[/yellow]")
            return

        # Step 1: Collect papers
        console.print(f"[bold cyan]Step 1: Collecting papers for '{query}'[/bold cyan]")

        arxiv_config = {
            "max_results": max_results,
            "rate_limit": 1.0
        }

        collector = ArXivCollector(arxiv_config)
        normalizer = AcademicNormalizer({})

        papers = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(f"Collecting papers...", total=None)

            async for result in collector.collect(
                search_query=query,
                category=category,
                start=0
            ):
                if result.success:
                    normalized = await normalizer.normalize(result)
                    papers.append(normalized)
                    progress.update(task, description=f"Collected {len(papers)} papers...")
                else:
                    console.print(f"[red]Collection error: {result.error_message}[/red]")

        if not papers:
            console.print("[yellow]No papers collected[/yellow]")
            return

        console.print(f"[green]✓ Collected {len(papers)} papers[/green]")

        # Step 2: Deep research analysis
        console.print(f"\n[bold cyan]Step 2: Performing deep research analysis[/bold cyan]")

        # Build configuration
        dr_config = build_deep_research_config_from_settings(config)
        dr_config.research_model = model  # Override with CLI option

        # Validate configuration
        validation_errors = validate_deep_research_config(dr_config)
        if validation_errors:
            console.print("[red]Configuration validation errors:[/red]")
            for field, error in validation_errors.items():
                console.print(f"  {field}: {error}")
            return

        # Setup storage manager
        storage_manager = ReportStorageManager()

        # Create bridge
        bridge = DeepResearchBridge(config=dr_config, storage_manager=storage_manager)

        # Perform analysis
        results = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Analyzing papers...", total=len(papers))

            for i, paper in enumerate(papers):
                try:
                    progress.update(task, description=f"Analyzing paper {i+1}/{len(papers)}: {paper.title[:50]}...")

                    result = await bridge.analyze_document(
                        document=paper,
                        research_question=research_question
                    )
                    results.append(result)

                    progress.update(task, advance=1)

                except Exception as e:
                    console.print(f"[red]Analysis failed for '{paper.title}': {e}[/red]")
                    continue

        console.print(f"[green]✓ Analyzed {len(results)} papers[/green]")

        # Step 3: Display results
        console.print(f"\n[bold cyan]Step 3: Analysis Results[/bold cyan]")

        for i, result in enumerate(results, 1):
            console.print(f"\n[bold yellow]Paper {i}: {result.metadata.get('document_title', 'Unknown')}[/bold yellow]")
            console.print(f"[bold]Research Question:[/bold] {result.research_question}")
            console.print(f"[bold]Analysis Summary:[/bold]")

            # Show first 500 chars of final report
            summary = result.final_report[:500]
            if len(result.final_report) > 500:
                summary += "..."
            console.print(summary)

            # Show key findings
            if result.research_findings:
                console.print(f"\n[bold]Key Findings:[/bold]")
                for finding in result.research_findings[:3]:  # Top 3 findings
                    console.print(f"• {finding[:200]}...")

        # Step 4: Save results if requested
        if output:
            output_path = Path(output)
            output_path.mkdir(parents=True, exist_ok=True)

            # Save detailed results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = output_path / f"deep_research_results_{timestamp}.json"

            # Convert results to JSON-serializable format
            json_results = []
            for result in results:
                json_results.append({
                    "document_id": result.document_id,
                    "research_question": result.research_question,
                    "final_report": result.final_report,
                    "research_findings": result.research_findings,
                    "metadata": result.metadata,
                    "timestamp": result.timestamp.isoformat(),
                    "model_used": result.model_used
                })

            with open(results_file, 'w') as f:
                json.dump(json_results, f, indent=2)

            console.print(f"\n[blue]Detailed results saved to: {results_file}[/blue]")

        # Show usage statistics
        stats = bridge.get_usage_statistics()
        console.print(f"\n[bold cyan]Analysis Statistics:[/bold cyan]")
        console.print(f"Success Rate: {stats['success_rate']:.2%}")
        console.print(f"Total Analyses: {stats['total_analyses']}")

    # Run the async analysis
    asyncio.run(_analyze())


@deep_research.command()
@click.argument("topic", required=True)
@click.option("--model", "-m", default="openai:gpt-4.1", help="Research model to use")
@click.option("--iterations", "-i", default=4, help="Max research iterations")
@click.option("--concurrent", "-c", default=3, help="Max concurrent research units")
@click.option("--search-api", "-s", default="tavily", help="Search API (tavily, openai, anthropic)")
@click.option("--output", "-o", help="Output file for report")
def comprehensive_analysis(topic: str, model: str, iterations: int, concurrent: int,
                          search_api: str, output: Optional[str]):
    """Perform comprehensive deep research analysis on any topic."""

    async def _analyze():
        # Load configuration
        config = load_config()

        if not is_deep_research_enabled(config):
            console.print("[red]Deep research is disabled in configuration[/red]")
            return

        console.print(f"[bold cyan]Starting comprehensive analysis: '{topic}'[/bold cyan]")

        # Build custom configuration
        dr_config = DeepResearchConfig(
            research_model=model,
            final_report_model=model,
            max_researcher_iterations=iterations,
            max_concurrent_research_units=concurrent,
            search_api=search_api,
            allow_clarification=False  # Disable for automated analysis
        )

        # Validate configuration
        validation_errors = validate_deep_research_config(dr_config)
        if validation_errors:
            console.print("[red]Configuration validation errors:[/red]")
            for field, error in validation_errors.items():
                console.print(f"  {field}: {error}")
            return

        # Create bridge with storage
        storage_manager = ReportStorageManager()
        bridge = DeepResearchBridge(config=dr_config, storage_manager=storage_manager)

        # Create a mock document for the topic
        from ..pipeline.data_types import NormalizedDocument
        from datetime import datetime

        mock_document = NormalizedDocument(
            id=f"comprehensive_{topic.replace(' ', '_')}",
            title=f"Comprehensive Analysis: {topic}",
            content=f"Research topic: {topic}",
            authors=[],
            published_date=datetime.now(),
            url="",
            categories=[],
            metadata={"analysis_type": "comprehensive"}
        )

        # Perform analysis
        with console.status(f"[bold green]Analyzing '{topic}'..."):
            result = await bridge.analyze_document(
                document=mock_document,
                research_question=f"Provide a comprehensive analysis of {topic}, including current state, key developments, challenges, and future outlook."
            )

        # Display results
        console.print(f"\n[bold green]✓ Analysis Complete[/bold green]")
        console.print(f"[bold]Topic:[/bold] {topic}")
        console.print(f"[bold]Model:[/bold] {result.model_used}")
        console.print(f"[bold]Analysis Date:[/bold] {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")

        console.print(f"\n[bold cyan]Research Report:[/bold cyan]")
        console.print(result.final_report)

        # Save if requested
        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            report_data = {
                "topic": topic,
                "research_question": result.research_question,
                "final_report": result.final_report,
                "research_findings": result.research_findings,
                "metadata": result.metadata,
                "timestamp": result.timestamp.isoformat(),
                "model_used": result.model_used,
                "configuration": {
                    "model": model,
                    "iterations": iterations,
                    "concurrent": concurrent,
                    "search_api": search_api
                }
            }

            with open(output_path, 'w') as f:
                json.dump(report_data, f, indent=2)

            console.print(f"\n[blue]Report saved to: {output_path}[/blue]")

    # Run the analysis
    asyncio.run(_analyze())


@deep_research.command()
def test_integration():
    """Test the OpenDeepResearch integration."""

    try:
        # Test imports
        console.print("[bold cyan]Testing OpenDeepResearch Integration[/bold cyan]")
        console.print("✓ Testing imports...")

        # Test configuration
        console.print("✓ Testing configuration...")
        config = load_config()

        if not is_deep_research_enabled(config):
            console.print("[yellow]Deep research is disabled in configuration[/yellow]")
        else:
            console.print("✓ Deep research is enabled")

        dr_config = build_deep_research_config_from_settings(config)
        validation_errors = validate_deep_research_config(dr_config)

        if validation_errors:
            console.print("[red]Configuration validation errors:[/red]")
            for field, error in validation_errors.items():
                console.print(f"  {field}: {error}")
        else:
            console.print("✓ Configuration is valid")

        # Test bridge creation
        console.print("✓ Testing bridge creation...")
        bridge = DeepResearchBridge(config=dr_config)
        console.print("✓ Bridge created successfully")

        # Show configuration details
        console.print(f"\n[bold cyan]Current Configuration:[/bold cyan]")
        console.print(f"Research Model: {dr_config.research_model}")
        console.print(f"Search API: {dr_config.search_api}")
        console.print(f"Max Iterations: {dr_config.max_researcher_iterations}")
        console.print(f"Max Concurrent Units: {dr_config.max_concurrent_research_units}")

        console.print(f"\n[green]✓ Integration test successful![/green]")

    except Exception as e:
        console.print(f"[red]Integration test failed: {e}[/red]")
        import traceback
        console.print(f"[red]{traceback.format_exc()}[/red]")


if __name__ == "__main__":
    main()