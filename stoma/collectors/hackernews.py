"""HackerNews collector for tech trends and discussions."""

import asyncio
import json
from datetime import datetime, timedelta
from typing import AsyncIterator, Dict, Any, List, Optional, Set
import aiohttp
from urllib.parse import urljoin

from .base import APICollector, CollectionResult, SourceType


class HackerNewsCollector(APICollector):
    """Collector for HackerNews stories and comments."""
    
    def __init__(self, config: Dict[str, Any]):
        # HackerNews API configuration
        if "base_url" not in config:
            config["base_url"] = "https://hacker-news.firebaseio.com/v0"
        
        super().__init__(config)
        
        # HackerNews specific configuration
        self.story_types = config.get("story_types", ["topstories", "newstories", "beststories"])
        self.max_stories = config.get("max_stories", 30)
        self.include_comments = config.get("include_comments", True)
        self.max_comments_per_story = config.get("max_comments_per_story", 10)
        self.min_score = config.get("min_score", 10)  # Minimum story score
        self.max_age_hours = config.get("max_age_hours", 24)  # Maximum story age
        
        # Keywords for filtering relevant stories
        self.tech_keywords = set([
            "ai", "ml", "machine learning", "deep learning", "neural", "algorithm",
            "data science", "python", "javascript", "react", "vue", "angular",
            "tensorflow", "pytorch", "research", "paper", "study", "analysis",
            "blockchain", "crypto", "bitcoin", "ethereum", "quantum", "robotics",
            "automation", "startup", "funding", "ipo", "acquisition", "api",
            "cloud", "aws", "azure", "google", "microsoft", "apple", "meta",
            "tesla", "spacex", "open source", "github", "programming", "developer",
            "software", "hardware", "security", "privacy", "database", "web3"
        ])
    
    def _get_source_type(self) -> SourceType:
        return SourceType.TECH_TRENDS
    
    async def collect(self, 
                     story_types: Optional[List[str]] = None,
                     hours_back: Optional[int] = None,
                     **kwargs) -> AsyncIterator[CollectionResult]:
        """
        Collect stories and comments from HackerNews.
        
        Args:
            story_types: Types of stories to collect (topstories, newstories, beststories)
            hours_back: How many hours back to collect stories
        """
        target_story_types = story_types or self.story_types
        max_age = hours_back or self.max_age_hours
        
        async with aiohttp.ClientSession() as session:
            # Track processed stories to avoid duplicates
            processed_stories: Set[int] = set()
            
            for story_type in target_story_types:
                try:
                    async for result in self._collect_story_type(
                        session, story_type, max_age, processed_stories
                    ):
                        yield result
                        # Rate limiting for HN API
                        await asyncio.sleep(1.0 / self.rate_limit)
                        
                except Exception as e:
                    yield CollectionResult(
                        source_id=f"hn_{story_type}_error",
                        source_type=self.source_type,
                        collected_at=datetime.now(),
                        data={},
                        metadata={"error": str(e), "story_type": story_type},
                        success=False,
                        error_message=f"Failed to collect {story_type}: {e}"
                    )
    
    async def _collect_story_type(self, 
                                session: aiohttp.ClientSession,
                                story_type: str,
                                max_age_hours: int,
                                processed_stories: Set[int]) -> AsyncIterator[CollectionResult]:
        """Collect stories of a specific type."""
        
        # Get list of story IDs
        url = f"{self.base_url}/{story_type}.json"
        
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    story_ids = await response.json()
                    
                    # Process stories (limited by max_stories)
                    for story_id in story_ids[:self.max_stories]:
                        if story_id in processed_stories:
                            continue
                        
                        processed_stories.add(story_id)
                        
                        # Get individual story
                        story_result = await self._get_story(session, story_id, max_age_hours)
                        if story_result:
                            yield story_result
                            
                            # Get comments if enabled and story is relevant
                            if (self.include_comments and 
                                story_result.data.get("descendants", 0) > 0 and
                                story_result.success):
                                
                                async for comment_result in self._collect_story_comments(
                                    session, story_id, story_result.data.get("title", "")
                                ):
                                    yield comment_result
                else:
                    raise Exception(f"HTTP {response.status}: {await response.text()}")
                    
        except Exception as e:
            raise Exception(f"Error collecting {story_type}: {e}")
    
    async def _get_story(self, 
                        session: aiohttp.ClientSession,
                        story_id: int,
                        max_age_hours: int) -> Optional[CollectionResult]:
        """Get individual story details."""
        
        url = f"{self.base_url}/item/{story_id}.json"
        
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    story_data = await response.json()
                    if story_data and story_data.get("type") == "story":
                        return await self._parse_hn_story(story_data, max_age_hours)
                return None
                
        except Exception as e:
            print(f"Error fetching story {story_id}: {e}")
            return None
    
    async def _parse_hn_story(self, story: Dict[str, Any], max_age_hours: int) -> Optional[CollectionResult]:
        """Parse a HackerNews story into a CollectionResult."""
        
        try:
            story_id = story.get("id")
            title = story.get("title", "")
            text = story.get("text", "")
            url = story.get("url", "")
            
            # Check story age
            story_time = story.get("time", 0)
            story_date = datetime.fromtimestamp(story_time) if story_time else datetime.now()
            age_hours = (datetime.now() - story_date).total_seconds() / 3600
            
            if age_hours > max_age_hours:
                return None
            
            # Check minimum score
            score = story.get("score", 0)
            if score < self.min_score:
                return None
            
            # Check relevance based on keywords
            full_text = f"{title} {text}".lower()
            if not self._is_relevant_story(full_text):
                return None
            
            # Combine title and text for content
            content = title
            if text:
                content += f"\n\n{text}"
            
            story_data = {
                "id": story_id,
                "title": title,
                "content": content,
                "text": text,
                "url": url,
                "hn_url": f"https://news.ycombinator.com/item?id={story_id}",
                "author": story.get("by", "unknown"),
                "score": score,
                "descendants": story.get("descendants", 0),  # Comment count
                "time": story_time,
                "created_date": story_date.isoformat(),
                "keywords": self._extract_keywords_from_story(title, text),
                "domain": self._extract_domain(url) if url else None,
                "story_type": self._classify_story_type(title, text, url)
            }
            
            return CollectionResult(
                source_id=f"hn_story_{story_id}",
                source_type=self.source_type,
                collected_at=datetime.now(),
                data=story_data,
                metadata={
                    "source": "hackernews",
                    "content_type": "story",
                    "score": score,
                    "descendants": story.get("descendants", 0)
                },
                raw_content=json.dumps(story, indent=2)
            )
            
        except Exception as e:
            print(f"Error parsing HN story: {e}")
            return None
    
    async def _collect_story_comments(self,
                                    session: aiohttp.ClientSession,
                                    story_id: int,
                                    story_title: str) -> AsyncIterator[CollectionResult]:
        """Collect top comments from a story."""
        
        # Get the story to access comment IDs
        url = f"{self.base_url}/item/{story_id}.json"
        
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    story_data = await response.json()
                    comment_ids = story_data.get("kids", [])
                    
                    # Get top comments (limited by max_comments_per_story)
                    for comment_id in comment_ids[:self.max_comments_per_story]:
                        comment_result = await self._get_comment(
                            session, comment_id, story_id, story_title
                        )
                        if comment_result:
                            yield comment_result
                            
        except Exception as e:
            print(f"Error collecting comments for story {story_id}: {e}")
    
    async def _get_comment(self,
                          session: aiohttp.ClientSession,
                          comment_id: int,
                          story_id: int,
                          story_title: str) -> Optional[CollectionResult]:
        """Get individual comment details."""
        
        url = f"{self.base_url}/item/{comment_id}.json"
        
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    comment_data = await response.json()
                    if (comment_data and 
                        comment_data.get("type") == "comment" and 
                        comment_data.get("text")):
                        
                        return await self._parse_hn_comment(
                            comment_data, story_id, story_title
                        )
                return None
                
        except Exception as e:
            print(f"Error fetching comment {comment_id}: {e}")
            return None
    
    async def _parse_hn_comment(self, 
                              comment: Dict[str, Any],
                              story_id: int,
                              story_title: str) -> Optional[CollectionResult]:
        """Parse a HackerNews comment into a CollectionResult."""
        
        try:
            comment_id = comment.get("id")
            text = comment.get("text", "")
            
            # Skip if no meaningful text
            if not text.strip():
                return None
            
            # Check if comment contains relevant content
            if not self._is_relevant_comment(text):
                return None
            
            comment_time = comment.get("time", 0)
            comment_date = datetime.fromtimestamp(comment_time) if comment_time else datetime.now()
            
            comment_data = {
                "id": comment_id,
                "content": text,
                "story_id": story_id,
                "story_title": story_title,
                "author": comment.get("by", "unknown"),
                "time": comment_time,
                "created_date": comment_date.isoformat(),
                "parent_id": comment.get("parent"),
                "hn_url": f"https://news.ycombinator.com/item?id={comment_id}",
                "keywords": self._extract_keywords_from_text(text),
                "content_type": "comment"
            }
            
            return CollectionResult(
                source_id=f"hn_comment_{comment_id}",
                source_type=self.source_type,
                collected_at=datetime.now(),
                data=comment_data,
                metadata={
                    "source": "hackernews",
                    "content_type": "comment",
                    "story_id": story_id
                },
                raw_content=json.dumps(comment, indent=2)
            )
            
        except Exception as e:
            print(f"Error parsing HN comment: {e}")
            return None
    
    def _is_relevant_story(self, text: str) -> bool:
        """Check if a story is relevant based on keywords."""
        text_lower = text.lower()
        
        # Check for tech keywords
        for keyword in self.tech_keywords:
            if keyword in text_lower:
                return True
        
        # Check for common patterns that indicate tech relevance
        tech_patterns = [
            "open source", "machine learning", "deep learning", "artificial intelligence",
            "data science", "programming", "software development", "web development",
            "mobile development", "cloud computing", "cybersecurity", "blockchain",
            "cryptocurrency", "startup", "tech company", "api", "framework",
            "library", "algorithm", "database", "architecture", "devops"
        ]
        
        for pattern in tech_patterns:
            if pattern in text_lower:
                return True
        
        return False
    
    def _is_relevant_comment(self, text: str) -> bool:
        """Check if a comment is relevant and substantial."""
        text_lower = text.lower()
        
        # Skip very short comments
        if len(text.split()) < 10:
            return False
        
        # Check for technical content
        return self._is_relevant_story(text_lower)
    
    def _extract_keywords_from_story(self, title: str, text: str) -> List[str]:
        """Extract keywords from HN story content."""
        keywords = []
        full_text = f"{title} {text}".lower()
        
        # Extract matching tech keywords
        for keyword in self.tech_keywords:
            if keyword in full_text:
                keywords.append(keyword)
        
        return list(set(keywords))
    
    def _extract_keywords_from_text(self, text: str) -> List[str]:
        """Extract keywords from text content."""
        keywords = []
        text_lower = text.lower()
        
        # Extract matching tech keywords
        for keyword in self.tech_keywords:
            if keyword in text_lower:
                keywords.append(keyword)
        
        return keywords
    
    def _extract_domain(self, url: str) -> Optional[str]:
        """Extract domain from URL."""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            return parsed.netloc
        except:
            return None
    
    def _classify_story_type(self, title: str, text: str, url: str) -> str:
        """Classify the type of HN story."""
        title_lower = title.lower()
        
        if title_lower.startswith("ask hn:"):
            return "ask_hn"
        elif title_lower.startswith("show hn:"):
            return "show_hn"
        elif title_lower.startswith("tell hn:"):
            return "tell_hn"
        elif url and any(domain in url for domain in ["github.com", "gitlab.com"]):
            return "open_source"
        elif any(word in title_lower for word in ["paper", "research", "study"]):
            return "research"
        elif any(word in title_lower for word in ["startup", "funding", "ipo", "acquisition"]):
            return "business"
        else:
            return "general"
    
    async def health_check(self) -> bool:
        """Check if HackerNews API is accessible."""
        try:
            async with aiohttp.ClientSession() as session:
                # Test with a simple request to get top stories
                url = f"{self.base_url}/topstories.json"
                async with session.get(url) as response:
                    return response.status == 200 and await response.json()
        except Exception:
            return False
    
    async def get_story_categories(self) -> List[str]:
        """Get available story categories."""
        return [
            "topstories",    # Top stories
            "newstories",    # New stories  
            "beststories",   # Best stories
            "askstories",    # Ask HN stories
            "showstories",   # Show HN stories
            "jobstories"     # Job stories
        ]
    
    async def search_stories_by_keyword(self, 
                                      keyword: str,
                                      max_results: int = 20) -> AsyncIterator[CollectionResult]:
        """Search for stories containing specific keywords."""
        
        async with aiohttp.ClientSession() as session:
            # Get recent stories and filter by keyword
            url = f"{self.base_url}/newstories.json"
            
            try:
                async with session.get(url) as response:
                    if response.status == 200:
                        story_ids = await response.json()
                        found_results = 0
                        
                        for story_id in story_ids:
                            if found_results >= max_results:
                                break
                                
                            story_result = await self._get_story(session, story_id, 48)  # 48 hours
                            if (story_result and 
                                keyword.lower() in story_result.data.get("content", "").lower()):
                                
                                yield story_result
                                found_results += 1
                                
            except Exception as e:
                yield CollectionResult(
                    source_id=f"hn_search_error",
                    source_type=self.source_type,
                    collected_at=datetime.now(),
                    data={},
                    metadata={"error": str(e), "keyword": keyword},
                    success=False,
                    error_message=f"Failed to search HN: {e}"
                )