"""Normalizer for code projects and repositories."""

from datetime import datetime
from typing import List, Dict, Any, Optional

from .base import BaseNormalizer, NormalizedData
from ..collectors.base import CollectionResult, SourceType


class CodeProjectsNormalizer(BaseNormalizer):
    """Normalizer for code projects, repositories, issues, and PRs."""
    
    def _get_supported_source_types(self) -> List[SourceType]:
        return [SourceType.CODE_PROJECTS]
    
    async def normalize(self, result: CollectionResult) -> NormalizedData:
        """Normalize code project data based on type."""
        
        data = result.data
        metadata_type = result.metadata.get("type", "repository")
        
        if metadata_type == "repository" or metadata_type == "repository_search":
            return await self._normalize_repository(result)
        elif metadata_type == "issue":
            return await self._normalize_issue(result)
        elif metadata_type == "pull_request":
            return await self._normalize_pull_request(result)
        elif metadata_type == "user_event":
            return await self._normalize_user_event(result)
        else:
            # Default to repository normalization
            return await self._normalize_repository(result)
    
    async def _normalize_repository(self, result: CollectionResult) -> NormalizedData:
        """Normalize repository data."""
        data = result.data
        
        # Create comprehensive content from repository information
        content_parts = []
        
        if data.get("description"):
            content_parts.append(f"Description: {data['description']}")
        
        if data.get("readme_content"):
            content_parts.append(f"README: {data['readme_content']}")
        
        # Add language information
        languages = data.get("languages", {})
        if languages:
            lang_list = ", ".join(languages.keys())
            content_parts.append(f"Languages: {lang_list}")
        
        # Add topics
        topics = data.get("topics", [])
        if topics:
            content_parts.append(f"Topics: {', '.join(topics)}")
        
        content = " | ".join(content_parts)
        
        # Generate summary
        summary = data.get("description", "")
        if not summary and content:
            summary = self._generate_summary(content, max_length=150)
        
        # Extract metrics
        metrics = {
            "stars": data.get("stargazers_count", 0),
            "watchers": data.get("watchers_count", 0),
            "forks": data.get("forks_count", 0),
            "open_issues": data.get("open_issues_count", 0),
            "size_kb": data.get("size", 0)
        }
        
        # Add language percentages if available
        if languages:
            total_bytes = sum(languages.values())
            if total_bytes > 0:
                metrics["language_percentages"] = {
                    lang: (bytes_count / total_bytes) * 100
                    for lang, bytes_count in languages.items()
                }
        
        # Determine categories
        categories = []
        main_language = data.get("language")
        if main_language:
            categories.append(main_language.lower())
        
        # Add categories based on topics
        if topics:
            categories.extend([topic.lower() for topic in topics[:5]])  # Limit topics
        
        # Add framework/library detection
        categories.extend(self._detect_frameworks(data))
        
        return NormalizedData(
            id=self._generate_repo_id(result),
            source_type=result.source_type,
            source_id=result.source_id,
            title=data.get("full_name", data.get("name", "")),
            content=content,
            summary=summary,
            authors=[data.get("owner", "")],
            published_date=self._parse_date(data.get("created_at")),
            collected_date=result.collected_at,
            url=data.get("html_url"),
            keywords=topics + [main_language] if main_language else topics,
            categories=list(set(categories)),  # Remove duplicates
            tags=self._generate_repo_tags(data),
            metrics=metrics,
            raw_data=data
        )
    
    async def _normalize_issue(self, result: CollectionResult) -> NormalizedData:
        """Normalize GitHub issue data."""
        data = result.data
        
        content = f"Issue: {data.get('body', '')}"
        
        return NormalizedData(
            id=self._generate_issue_id(result),
            source_type=result.source_type,
            source_id=result.source_id,
            title=f"Issue #{data.get('number')}: {data.get('title', '')}",
            content=content,
            summary=self._generate_summary(content, max_length=200),
            authors=[data.get("user", "")],
            published_date=self._parse_date(data.get("created_at")),
            collected_date=result.collected_at,
            url=data.get("html_url"),
            keywords=data.get("labels", []),
            categories=["issue", "github"],
            tags=["issue", data.get("state", "")],
            metrics={
                "comments": data.get("comments", 0),
                "number": data.get("number", 0)
            },
            raw_data=data
        )
    
    async def _normalize_pull_request(self, result: CollectionResult) -> NormalizedData:
        """Normalize GitHub pull request data."""
        data = result.data
        
        content = f"Pull Request: {data.get('body', '')}"
        
        return NormalizedData(
            id=self._generate_pr_id(result),
            source_type=result.source_type,
            source_id=result.source_id,
            title=f"PR #{data.get('number')}: {data.get('title', '')}",
            content=content,
            summary=self._generate_summary(content, max_length=200),
            authors=[data.get("user", "")],
            published_date=self._parse_date(data.get("created_at")),
            collected_date=result.collected_at,
            url=data.get("html_url"),
            keywords=[data.get("head_branch", ""), data.get("base_branch", "")],
            categories=["pull_request", "github"],
            tags=["pr", data.get("state", "")],
            metrics={
                "number": data.get("number", 0),
                "mergeable": data.get("mergeable"),
                "draft": data.get("draft", False)
            },
            raw_data=data
        )
    
    async def _normalize_user_event(self, result: CollectionResult) -> NormalizedData:
        """Normalize GitHub user event data."""
        data = result.data
        
        event_type = data.get("type", "")
        repo_name = data.get("repo", "")
        actor = data.get("actor", "")
        
        title = f"{actor} {event_type} on {repo_name}"
        content = f"GitHub event: {event_type} by {actor} on repository {repo_name}"
        
        return NormalizedData(
            id=self._generate_event_id(result),
            source_type=result.source_type,
            source_id=result.source_id,
            title=title,
            content=content,
            summary=content,
            authors=[actor],
            published_date=self._parse_date(data.get("created_at")),
            collected_date=result.collected_at,
            url=f"https://github.com/{repo_name}",
            keywords=[event_type, repo_name.split("/")[-1] if "/" in repo_name else repo_name],
            categories=["github_event", event_type.lower()],
            tags=["event", event_type.lower()],
            metrics={"event_type": event_type},
            raw_data=data
        )
    
    def _generate_repo_id(self, result: CollectionResult) -> str:
        """Generate unique ID for repositories."""
        repo_id = result.data.get("id", "")
        full_name = result.data.get("full_name", "")
        return f"github_repo_{repo_id}_{full_name}".replace("/", "_")
    
    def _generate_issue_id(self, result: CollectionResult) -> str:
        """Generate unique ID for issues."""
        issue_id = result.data.get("id", "")
        return f"github_issue_{issue_id}"
    
    def _generate_pr_id(self, result: CollectionResult) -> str:
        """Generate unique ID for pull requests."""
        pr_id = result.data.get("id", "")
        return f"github_pr_{pr_id}"
    
    def _generate_event_id(self, result: CollectionResult) -> str:
        """Generate unique ID for events."""
        event_id = result.data.get("id", "")
        return f"github_event_{event_id}"
    
    def _detect_frameworks(self, repo_data: Dict[str, Any]) -> List[str]:
        """Detect frameworks and libraries from repository data."""
        frameworks = []
        
        # Framework detection based on topics and description
        description = ((repo_data.get("description") or "") + " " + 
                      " ".join(repo_data.get("topics", []))).lower()
        
        framework_keywords = {
            "react": ["react", "jsx"],
            "vue": ["vue", "vuejs"],
            "angular": ["angular", "ng"],
            "django": ["django"],
            "flask": ["flask"],
            "express": ["express", "expressjs"],
            "spring": ["spring", "springboot"],
            "rails": ["rails", "ruby-on-rails"],
            "laravel": ["laravel"],
            "tensorflow": ["tensorflow", "tf"],
            "pytorch": ["pytorch", "torch"],
            "scikit-learn": ["scikit-learn", "sklearn"],
            "pandas": ["pandas"],
            "numpy": ["numpy"],
            "docker": ["docker", "dockerfile", "containerization"],
            "kubernetes": ["kubernetes", "k8s"],
            "aws": ["aws", "amazon-web-services"],
            "azure": ["azure", "microsoft-azure"],
            "gcp": ["gcp", "google-cloud"]
        }
        
        for framework, keywords in framework_keywords.items():
            if any(keyword in description for keyword in keywords):
                frameworks.append(framework)
        
        return frameworks
    
    def _generate_repo_tags(self, repo_data: Dict[str, Any]) -> List[str]:
        """Generate tags for repository."""
        tags = []
        
        # Add basic tags
        tags.append("repository")
        
        if repo_data.get("is_fork"):
            tags.append("fork")
        
        if repo_data.get("is_archived"):
            tags.append("archived")
        
        # Add license tag
        license_name = repo_data.get("license")
        if license_name:
            tags.append(f"license-{license_name.lower().replace(' ', '-')}")
        
        # Add popularity tags based on stars
        stars = repo_data.get("stargazers_count", 0)
        if stars > 10000:
            tags.append("popular")
        elif stars > 1000:
            tags.append("trending")
        elif stars > 100:
            tags.append("growing")
        
        # Add activity tags based on recent updates
        updated_at = repo_data.get("updated_at")
        if updated_at:
            try:
                updated_date = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
                days_since_update = (datetime.now(updated_date.tzinfo) - updated_date).days
                
                if days_since_update < 7:
                    tags.append("recently-updated")
                elif days_since_update < 30:
                    tags.append("active")
                elif days_since_update > 365:
                    tags.append("stale")
            except Exception:
                pass
        
        return tags
    
    def _parse_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse ISO date format from GitHub API."""
        if not date_str:
            return None
        
        try:
            # GitHub returns ISO format with Z suffix
            return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except Exception:
            return None