"""GitHub collector for code projects and repositories."""

import asyncio
import base64
from datetime import datetime, timedelta
from typing import AsyncIterator, Dict, Any, Optional, List
import aiohttp

from .base import APICollector, CollectionResult, SourceType


class GitHubCollector(APICollector):
    """Collector for GitHub repositories, issues, and trends."""
    
    def __init__(self, config: Dict[str, Any]):
        if "base_url" not in config:
            config["base_url"] = "https://api.github.com"
        
        super().__init__(config)
        
        # GitHub specific configuration
        self.token = config.get("token")  # GitHub API token
        self.per_page = config.get("per_page", 30)
        self.max_pages = config.get("max_pages", 3)
        
        # Set up headers
        if self.token:
            self.headers["Authorization"] = f"token {self.token}"
        self.headers["Accept"] = "application/vnd.github.v3+json"
        self.headers["User-Agent"] = "Stoma/1.0 (Research Intelligence System)"
    
    def _get_source_type(self) -> SourceType:
        return SourceType.CODE_PROJECTS
    
    async def collect(self, **kwargs) -> AsyncIterator[CollectionResult]:
        """Default collect method - delegates to trending repos."""
        async for result in self.collect_trending_repos(**kwargs):
            yield result
    
    async def collect_trending_repos(self, 
                                   language: str = None,
                                   created_since: str = "week",
                                   **kwargs) -> AsyncIterator[CollectionResult]:
        """
        Collect trending repositories.
        
        Args:
            language: Programming language filter (e.g., "python", "javascript")
            created_since: Time period ("week", "month", "year")
        """
        
        # Build search query for trending repos
        date_cutoff = self._get_date_cutoff(created_since)
        query_parts = [f"created:>{date_cutoff}"]
        
        if language:
            query_parts.append(f"language:{language}")
        
        query = " ".join(query_parts)
        
        async for result in self._search_repositories(
            query=query,
            sort="stars",
            order="desc"
        ):
            yield result
    
    async def collect_repo_info(self, owner: str, repo: str) -> AsyncIterator[CollectionResult]:
        """Collect detailed information about a specific repository."""
        
        endpoint = f"/repos/{owner}/{repo}"
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(
                    f"{self.base_url}{endpoint}",
                    headers=self.headers
                ) as response:
                    if response.status == 200:
                        repo_data = await response.json()
                        
                        # Also collect additional data
                        additional_data = await self._collect_repo_additional_data(
                            session, owner, repo
                        )
                        repo_data.update(additional_data)
                        
                        yield CollectionResult(
                            source_id=f"github_repo_{owner}_{repo}",
                            source_type=self.source_type,
                            collected_at=datetime.now(),
                            data=self._normalize_repo_data(repo_data),
                            metadata={
                                "source": "github",
                                "type": "repository",
                                "owner": owner,
                                "repo": repo
                            }
                        )
                    else:
                        yield self._create_error_result(
                            f"github_repo_error_{owner}_{repo}",
                            f"Failed to fetch repository: HTTP {response.status}"
                        )
                        
            except Exception as e:
                yield self._create_error_result(
                    f"github_repo_exception_{owner}_{repo}",
                    f"Exception while collecting repository: {e}"
                )
    
    async def collect_issues_and_prs(self, 
                                   owner: str, 
                                   repo: str,
                                   state: str = "open",
                                   labels: List[str] = None) -> AsyncIterator[CollectionResult]:
        """Collect issues and pull requests from a repository."""
        
        # Collect issues
        async for result in self._collect_issues(owner, repo, state, labels):
            yield result
        
        # Collect pull requests
        async for result in self._collect_pull_requests(owner, repo, state):
            yield result
    
    async def collect_user_activity(self, username: str) -> AsyncIterator[CollectionResult]:
        """Collect activity for a specific user."""
        
        endpoint = f"/users/{username}/events/public"
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(
                    f"{self.base_url}{endpoint}",
                    headers=self.headers
                ) as response:
                    if response.status == 200:
                        events = await response.json()
                        
                        for event in events[:20]:  # Limit to recent events
                            yield CollectionResult(
                                source_id=f"github_event_{event.get('id')}",
                                source_type=self.source_type,
                                collected_at=datetime.now(),
                                data=self._normalize_event_data(event),
                                metadata={
                                    "source": "github",
                                    "type": "user_event",
                                    "username": username
                                }
                            )
                    else:
                        yield self._create_error_result(
                            f"github_user_error_{username}",
                            f"Failed to fetch user activity: HTTP {response.status}"
                        )
                        
            except Exception as e:
                yield self._create_error_result(
                    f"github_user_exception_{username}",
                    f"Exception while collecting user activity: {e}"
                )
    
    async def _search_repositories(self, 
                                 query: str,
                                 sort: str = "stars",
                                 order: str = "desc") -> AsyncIterator[CollectionResult]:
        """Search repositories using GitHub search API."""
        
        endpoint = "/search/repositories"
        
        async with aiohttp.ClientSession() as session:
            for page in range(1, self.max_pages + 1):
                try:
                    params = {
                        "q": query,
                        "sort": sort,
                        "order": order,
                        "per_page": self.per_page,
                        "page": page
                    }
                    
                    async with session.get(
                        f"{self.base_url}{endpoint}",
                        headers=self.headers,
                        params=params
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            for repo in data.get("items", []):
                                yield CollectionResult(
                                    source_id=f"github_search_{repo['id']}",
                                    source_type=self.source_type,
                                    collected_at=datetime.now(),
                                    data=self._normalize_repo_data(repo),
                                    metadata={
                                        "source": "github",
                                        "type": "repository_search",
                                        "query": query,
                                        "page": page
                                    }
                                )
                            
                            # Rate limiting
                            await asyncio.sleep(1.0 / self.rate_limit)
                            
                            # Stop if we've exhausted results
                            if len(data.get("items", [])) < self.per_page:
                                break
                                
                        elif response.status == 403:
                            # Rate limit exceeded
                            yield self._create_error_result(
                                "github_rate_limit",
                                "GitHub API rate limit exceeded"
                            )
                            break
                        else:
                            yield self._create_error_result(
                                f"github_search_error_page_{page}",
                                f"Search failed: HTTP {response.status}"
                            )
                            
                except Exception as e:
                    yield self._create_error_result(
                        f"github_search_exception_page_{page}",
                        f"Exception during search: {e}"
                    )
    
    async def _collect_repo_additional_data(self, 
                                          session: aiohttp.ClientSession,
                                          owner: str, 
                                          repo: str) -> Dict[str, Any]:
        """Collect additional repository data (README, languages, etc.)."""
        
        additional_data = {}
        
        try:
            # Get languages
            async with session.get(
                f"{self.base_url}/repos/{owner}/{repo}/languages",
                headers=self.headers
            ) as response:
                if response.status == 200:
                    additional_data["languages"] = await response.json()
        
            # Get README
            async with session.get(
                f"{self.base_url}/repos/{owner}/{repo}/readme",
                headers=self.headers
            ) as response:
                if response.status == 200:
                    readme_data = await response.json()
                    if readme_data.get("content"):
                        # Decode base64 content
                        content = base64.b64decode(readme_data["content"]).decode("utf-8")
                        additional_data["readme_content"] = content[:2000]  # Limit size
            
            # Get recent releases
            async with session.get(
                f"{self.base_url}/repos/{owner}/{repo}/releases",
                headers=self.headers,
                params={"per_page": 5}
            ) as response:
                if response.status == 200:
                    additional_data["recent_releases"] = await response.json()
        
        except Exception:
            # Don't fail if additional data collection fails
            pass
        
        return additional_data
    
    async def _collect_issues(self, 
                            owner: str, 
                            repo: str, 
                            state: str,
                            labels: List[str] = None) -> AsyncIterator[CollectionResult]:
        """Collect issues from a repository."""
        
        endpoint = f"/repos/{owner}/{repo}/issues"
        
        async with aiohttp.ClientSession() as session:
            params = {
                "state": state,
                "per_page": self.per_page
            }
            
            if labels:
                params["labels"] = ",".join(labels)
            
            try:
                async with session.get(
                    f"{self.base_url}{endpoint}",
                    headers=self.headers,
                    params=params
                ) as response:
                    if response.status == 200:
                        issues = await response.json()
                        
                        for issue in issues:
                            # Skip pull requests (they appear in issues API)
                            if "pull_request" not in issue:
                                yield CollectionResult(
                                    source_id=f"github_issue_{issue['id']}",
                                    source_type=self.source_type,
                                    collected_at=datetime.now(),
                                    data=self._normalize_issue_data(issue),
                                    metadata={
                                        "source": "github",
                                        "type": "issue",
                                        "owner": owner,
                                        "repo": repo
                                    }
                                )
            except Exception as e:
                yield self._create_error_result(
                    f"github_issues_exception_{owner}_{repo}",
                    f"Exception while collecting issues: {e}"
                )
    
    async def _collect_pull_requests(self, 
                                   owner: str, 
                                   repo: str, 
                                   state: str) -> AsyncIterator[CollectionResult]:
        """Collect pull requests from a repository."""
        
        endpoint = f"/repos/{owner}/{repo}/pulls"
        
        async with aiohttp.ClientSession() as session:
            params = {
                "state": state,
                "per_page": self.per_page
            }
            
            try:
                async with session.get(
                    f"{self.base_url}{endpoint}",
                    headers=self.headers,
                    params=params
                ) as response:
                    if response.status == 200:
                        prs = await response.json()
                        
                        for pr in prs:
                            yield CollectionResult(
                                source_id=f"github_pr_{pr['id']}",
                                source_type=self.source_type,
                                collected_at=datetime.now(),
                                data=self._normalize_pr_data(pr),
                                metadata={
                                    "source": "github",
                                    "type": "pull_request",
                                    "owner": owner,
                                    "repo": repo
                                }
                            )
            except Exception as e:
                yield self._create_error_result(
                    f"github_prs_exception_{owner}_{repo}",
                    f"Exception while collecting pull requests: {e}"
                )
    
    def _normalize_repo_data(self, repo: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize repository data."""
        return {
            "id": repo.get("id"),
            "name": repo.get("name"),
            "full_name": repo.get("full_name"),
            "description": repo.get("description", ""),
            "owner": repo.get("owner", {}).get("login"),
            "html_url": repo.get("html_url"),
            "clone_url": repo.get("clone_url"),
            "language": repo.get("language"),
            "languages": repo.get("languages", {}),
            "stargazers_count": repo.get("stargazers_count", 0),
            "watchers_count": repo.get("watchers_count", 0),
            "forks_count": repo.get("forks_count", 0),
            "open_issues_count": repo.get("open_issues_count", 0),
            "created_at": repo.get("created_at"),
            "updated_at": repo.get("updated_at"),
            "pushed_at": repo.get("pushed_at"),
            "size": repo.get("size"),
            "topics": repo.get("topics", []),
            "license": repo.get("license", {}).get("name") if repo.get("license") else None,
            "readme_content": repo.get("readme_content"),
            "recent_releases": repo.get("recent_releases", []),
            "is_fork": repo.get("fork", False),
            "is_archived": repo.get("archived", False),
            "default_branch": repo.get("default_branch")
        }
    
    def _normalize_issue_data(self, issue: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize issue data."""
        return {
            "id": issue.get("id"),
            "number": issue.get("number"),
            "title": issue.get("title"),
            "body": issue.get("body", "")[:1000],  # Limit body size
            "state": issue.get("state"),
            "user": issue.get("user", {}).get("login"),
            "assignee": issue.get("assignee", {}).get("login") if issue.get("assignee") else None,
            "labels": [label.get("name") for label in issue.get("labels", [])],
            "created_at": issue.get("created_at"),
            "updated_at": issue.get("updated_at"),
            "closed_at": issue.get("closed_at"),
            "html_url": issue.get("html_url"),
            "comments": issue.get("comments", 0)
        }
    
    def _normalize_pr_data(self, pr: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize pull request data."""
        return {
            "id": pr.get("id"),
            "number": pr.get("number"),
            "title": pr.get("title"),
            "body": pr.get("body", "")[:1000],  # Limit body size
            "state": pr.get("state"),
            "user": pr.get("user", {}).get("login"),
            "assignee": pr.get("assignee", {}).get("login") if pr.get("assignee") else None,
            "head_branch": pr.get("head", {}).get("ref"),
            "base_branch": pr.get("base", {}).get("ref"),
            "created_at": pr.get("created_at"),
            "updated_at": pr.get("updated_at"),
            "closed_at": pr.get("closed_at"),
            "merged_at": pr.get("merged_at"),
            "html_url": pr.get("html_url"),
            "mergeable": pr.get("mergeable"),
            "draft": pr.get("draft", False)
        }
    
    def _normalize_event_data(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize user event data."""
        return {
            "id": event.get("id"),
            "type": event.get("type"),
            "actor": event.get("actor", {}).get("login"),
            "repo": event.get("repo", {}).get("name"),
            "created_at": event.get("created_at"),
            "payload": event.get("payload", {})
        }
    
    def _get_date_cutoff(self, period: str) -> str:
        """Get date cutoff for trending repos."""
        now = datetime.now()
        
        if period == "week":
            cutoff = now - timedelta(weeks=1)
        elif period == "month":
            cutoff = now - timedelta(days=30)
        elif period == "year":
            cutoff = now - timedelta(days=365)
        else:
            cutoff = now - timedelta(weeks=1)  # Default to week
        
        return cutoff.strftime("%Y-%m-%d")
    
    def _create_error_result(self, source_id: str, error_message: str) -> CollectionResult:
        """Create an error result."""
        return CollectionResult(
            source_id=source_id,
            source_type=self.source_type,
            collected_at=datetime.now(),
            data={},
            metadata={"error": error_message},
            success=False,
            error_message=error_message
        )
    
    async def health_check(self) -> bool:
        """Check if GitHub API is accessible."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/user",
                    headers=self.headers
                ) as response:
                    # 200 if authenticated, 401 if not (but API is accessible)
                    return response.status in [200, 401]
        except Exception:
            return False