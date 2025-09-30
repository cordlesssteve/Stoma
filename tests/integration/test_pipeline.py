#!/usr/bin/env python3
"""
Comprehensive test of the Stoma data pipeline.
This script tests the complete flow from collection to storage.
"""

import asyncio
import sys
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from stoma.collectors.arxiv import ArXivCollector
from stoma.collectors.github import GitHubCollector
from stoma.normalizers.base import AcademicNormalizer
from stoma.normalizers.code_projects import CodeProjectsNormalizer
from stoma.storage.base import PostgreSQLStorage
from stoma.config.settings import load_config

console = Console()


async def test_arxiv_pipeline():
    """Test ArXiv collection, normalization, and storage."""
    console.print("\n[bold blue]Testing ArXiv Pipeline[/bold blue]")
    
    # Setup
    collector = ArXivCollector({"max_results": 3, "rate_limit": 1.0})
    normalizer = AcademicNormalizer({})
    
    results = []
    
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
        task = progress.add_task("Collecting ArXiv papers...", total=None)
        
        async for result in collector.collect(search_query="machine learning", start=0):
            if result.success:
                normalized = await normalizer.normalize(result)
                results.append(normalized)
                progress.update(task, description=f"Collected {len(results)} papers...")
            else:
                console.print(f"[red]Error: {result.error_message}[/red]")
            
            if len(results) >= 3:
                break
    
    # Display results
    if results:
        table = Table(title="ArXiv Papers Collected")
        table.add_column("Title", style="cyan", max_width=50)
        table.add_column("Authors", style="magenta", max_width=30)
        table.add_column("Published", style="green")
        
        for paper in results:
            authors_str = ", ".join(paper.authors[:2])
            if len(paper.authors) > 2:
                authors_str += f" (+{len(paper.authors) - 2} more)"
            
            published = paper.published_date.strftime("%Y-%m-%d") if paper.published_date else "Unknown"
            
            table.add_row(
                paper.title[:80] + "..." if len(paper.title) > 80 else paper.title,
                authors_str,
                published
            )
        
        console.print(table)
    
    return results


async def test_github_pipeline():
    """Test GitHub collection, normalization, and storage."""
    console.print("\n[bold blue]Testing GitHub Pipeline[/bold blue]")
    
    # Setup
    collector = GitHubCollector({"per_page": 3, "max_pages": 1, "rate_limit": 1.0})
    normalizer = CodeProjectsNormalizer({})
    
    results = []
    
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
        task = progress.add_task("Collecting GitHub repos...", total=None)
        
        async for result in collector.collect_trending_repos(language="python", created_since="week"):
            if result.success:
                normalized = await normalizer.normalize(result)
                results.append(normalized)
                progress.update(task, description=f"Collected {len(results)} repositories...")
            else:
                console.print(f"[red]Error: {result.error_message}[/red]")
            
            if len(results) >= 3:
                break
    
    # Display results
    if results:
        table = Table(title="GitHub Repositories Collected")
        table.add_column("Repository", style="cyan", max_width=40)
        table.add_column("Language", style="magenta")
        table.add_column("Stars", style="yellow")
        table.add_column("Description", style="white", max_width=50)
        
        for repo in results:
            stars = repo.metrics.get("stars", 0)
            language = next((cat for cat in repo.categories if cat not in ["repository", "python"]), "Unknown")
            summary = repo.summary or ""
            
            table.add_row(
                repo.title,
                language.title(),
                str(stars),
                summary[:100] + "..." if len(summary) > 100 else summary
            )
        
        console.print(table)
    
    return results


async def test_storage_pipeline(arxiv_results, github_results):
    """Test storing data in PostgreSQL."""
    console.print("\n[bold blue]Testing Storage Pipeline[/bold blue]")
    
    # Load configuration
    try:
        config = load_config("config.yaml")
        storage_config = config["storage"]
    except Exception as e:
        console.print(f"[red]Error loading config: {e}[/red]")
        console.print("[yellow]Skipping storage test[/yellow]")
        return False
    
    # Setup storage
    storage = PostgreSQLStorage(storage_config)
    
    try:
        await storage.connect()
        console.print("[green]✓ Connected to database[/green]")
        
        stored_count = 0
        
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
            task = progress.add_task("Storing data...", total=None)
            
            # Store ArXiv papers
            for paper in arxiv_results:
                try:
                    storage_id = await storage.store(paper)
                    stored_count += 1
                    progress.update(task, description=f"Stored {stored_count} items...")
                except Exception as e:
                    console.print(f"[red]Error storing ArXiv paper: {e}[/red]")
            
            # Store GitHub repos
            for repo in github_results:
                try:
                    storage_id = await storage.store(repo)
                    stored_count += 1
                    progress.update(task, description=f"Stored {stored_count} items...")
                except Exception as e:
                    console.print(f"[red]Error storing GitHub repo: {e}[/red]")
        
        console.print(f"[green]✓ Successfully stored {stored_count} items[/green]")
        
        # Test retrieval with search
        console.print("\n[bold]Testing Search Functionality[/bold]")
        search_results = []
        async for result in storage.search("machine learning", limit=5):
            search_results.append(result)
        
        if search_results:
            console.print(f"[green]✓ Search returned {len(search_results)} results[/green]")
            
            search_table = Table(title="Search Results: 'machine learning'")
            search_table.add_column("Source", style="cyan")
            search_table.add_column("Title", style="white", max_width=60)
            search_table.add_column("Type", style="magenta")
            
            for item in search_results[:3]:  # Show first 3
                source_type = item.source_type.value
                title = item.title[:80] + "..." if len(item.title) > 80 else item.title
                
                search_table.add_row(
                    source_type.replace("_", " ").title(),
                    title,
                    item.categories[0] if item.categories else "Unknown"
                )
            
            console.print(search_table)
        else:
            console.print("[yellow]No search results found[/yellow]")
        
        await storage.disconnect()
        console.print("[green]✓ Disconnected from database[/green]")
        return True
        
    except Exception as e:
        console.print(f"[red]Storage test failed: {e}[/red]")
        try:
            await storage.disconnect()
        except:
            pass
        return False


async def main():
    """Run comprehensive pipeline test."""
    console.print("[bold green]Stoma Comprehensive Pipeline Test[/bold green]")
    console.print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test ArXiv pipeline
    try:
        arxiv_results = await test_arxiv_pipeline()
        console.print(f"[green]✓ ArXiv pipeline: {len(arxiv_results)} items collected[/green]")
    except Exception as e:
        console.print(f"[red]✗ ArXiv pipeline failed: {e}[/red]")
        arxiv_results = []
    
    # Test GitHub pipeline
    try:
        github_results = await test_github_pipeline()
        console.print(f"[green]✓ GitHub pipeline: {len(github_results)} items collected[/green]")
    except Exception as e:
        console.print(f"[red]✗ GitHub pipeline failed: {e}[/red]")
        github_results = []
    
    # Test storage pipeline
    if arxiv_results or github_results:
        try:
            storage_success = await test_storage_pipeline(arxiv_results, github_results)
            if storage_success:
                console.print("[green]✓ Storage pipeline successful[/green]")
            else:
                console.print("[red]✗ Storage pipeline failed[/red]")
        except Exception as e:
            console.print(f"[red]✗ Storage pipeline error: {e}[/red]")
    else:
        console.print("[yellow]⚠ No data to test storage with[/yellow]")
    
    # Summary
    console.print(f"\n[bold]Test Summary[/bold]")
    console.print(f"ArXiv items: {len(arxiv_results)}")
    console.print(f"GitHub items: {len(github_results)}")
    console.print(f"Total items: {len(arxiv_results) + len(github_results)}")
    
    console.print(f"\n[bold green]Pipeline test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/bold green]")


if __name__ == "__main__":
    asyncio.run(main())