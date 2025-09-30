"""Reddit collector for social discussions and trends."""

import asyncio
import json
from datetime import datetime, timedelta
from typing import AsyncIterator, Dict, Any, List, Optional
import aiohttp
from urllib.parse import urljoin

from .base import APICollector, CollectionResult, SourceType


class RedditCollector(APICollector):
    """Collector for Reddit discussions and trending topics."""
    
    def __init__(self, config: Dict[str, Any]):
        # Reddit API configuration
        if "base_url" not in config:
            config["base_url"] = "https://www.reddit.com"
        
        super().__init__(config)
        
        # Reddit specific configuration
        self.subreddits = config.get("subreddits", [
            "MachineLearning", "science", "technology", "programming",
            "artificial", "deeplearning", "research", "datascience"
        ])
        self.sort_by = config.get("sort_by", "hot")  # hot, new, top, rising
        self.time_filter = config.get("time_filter", "day")  # hour, day, week, month, year, all
        self.max_posts = config.get("max_posts", 25)
        self.include_comments = config.get("include_comments", True)
        self.max_comments = config.get("max_comments", 10)
        
        # Set User-Agent for Reddit API
        self.headers.update({
            "User-Agent": "Stoma/1.0 (Research Intelligence System)"
        })
    
    def _get_source_type(self) -> SourceType:
        return SourceType.SOCIAL_DISCUSSIONS
    
    async def collect(self, 
                     subreddits: Optional[List[str]] = None,
                     time_range: Optional[str] = None,
                     **kwargs) -> AsyncIterator[CollectionResult]:
        """
        Collect posts and discussions from Reddit.
        
        Args:
            subreddits: List of subreddits to collect from (overrides config)
            time_range: Time range for collection (overrides config)
        """
        target_subreddits = subreddits or self.subreddits
        time_filter = time_range or self.time_filter
        
        async with aiohttp.ClientSession(headers=self.headers) as session:
            for subreddit in target_subreddits:
                try:
                    async for result in self._collect_from_subreddit(
                        session, subreddit, time_filter
                    ):
                        yield result
                        # Rate limiting for Reddit API
                        await asyncio.sleep(1.0 / self.rate_limit)
                        
                except Exception as e:
                    yield CollectionResult(
                        source_id=f"reddit_{subreddit}_error",
                        source_type=self.source_type,
                        collected_at=datetime.now(),
                        data={},
                        metadata={"error": str(e), "subreddit": subreddit},
                        success=False,
                        error_message=f"Failed to collect from r/{subreddit}: {e}"
                    )
    
    async def _collect_from_subreddit(self, 
                                    session: aiohttp.ClientSession,
                                    subreddit: str,
                                    time_filter: str) -> AsyncIterator[CollectionResult]:
        """Collect posts from a specific subreddit."""
        
        # Build Reddit JSON API URL
        url = f"{self.base_url}/r/{subreddit}/{self.sort_by}.json"
        params = {
            "limit": self.max_posts,
            "t": time_filter if self.sort_by == "top" else None
        }
        
        try:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    for post_data in data.get("data", {}).get("children", []):
                        post = post_data.get("data", {})
                        if not post:
                            continue
                        
                        # Extract post information
                        post_result = await self._parse_reddit_post(post, subreddit)
                        if post_result:
                            yield post_result
                        
                        # Collect comments if enabled
                        if self.include_comments and post.get("num_comments", 0) > 0:
                            async for comment_result in self._collect_post_comments(
                                session, subreddit, post.get("id"), post.get("title", "")
                            ):
                                yield comment_result
                                
                elif response.status == 429:
                    # Rate limited - wait longer
                    await asyncio.sleep(60)
                    raise Exception("Rate limited by Reddit API")
                else:
                    raise Exception(f"HTTP {response.status}: {await response.text()}")
                    
        except Exception as e:
            raise Exception(f"Error collecting from r/{subreddit}: {e}")
    
    async def _collect_post_comments(self,
                                   session: aiohttp.ClientSession,
                                   subreddit: str,
                                   post_id: str,
                                   post_title: str) -> AsyncIterator[CollectionResult]:
        """Collect comments from a specific post."""
        
        url = f"{self.base_url}/r/{subreddit}/comments/{post_id}.json"
        params = {"limit": self.max_comments, "sort": "top"}
        
        try:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Reddit returns [post_data, comments_data]
                    if len(data) >= 2:
                        comments_data = data[1]
                        for comment_data in comments_data.get("data", {}).get("children", []):
                            comment = comment_data.get("data", {})
                            if comment and comment.get("body"):
                                comment_result = await self._parse_reddit_comment(
                                    comment, subreddit, post_id, post_title
                                )
                                if comment_result:
                                    yield comment_result
                                    
        except Exception as e:
            # Don't fail the entire collection for comment errors
            print(f"Error collecting comments for post {post_id}: {e}")
    
    async def _parse_reddit_post(self, post: Dict[str, Any], subreddit: str) -> Optional[CollectionResult]:
        """Parse a Reddit post into a CollectionResult."""
        
        try:
            # Extract key information
            post_id = post.get("id")
            title = post.get("title", "")
            selftext = post.get("selftext", "")
            url = post.get("url", "")
            permalink = f"{self.base_url}{post.get('permalink', '')}"
            
            # Combine title and content
            content = title
            if selftext:
                content += f"\n\n{selftext}"
            
            # Skip if no meaningful content
            if not content.strip():
                return None
            
            # Extract metadata
            created_utc = post.get("created_utc", 0)
            created_date = datetime.fromtimestamp(created_utc) if created_utc else datetime.now()
            
            post_data = {
                "id": post_id,
                "title": title,
                "content": content,
                "selftext": selftext,
                "url": url,
                "permalink": permalink,
                "subreddit": subreddit,
                "author": post.get("author", "[deleted]"),
                "score": post.get("score", 0),
                "upvote_ratio": post.get("upvote_ratio", 0.0),
                "num_comments": post.get("num_comments", 0),
                "created_utc": created_utc,
                "created_date": created_date.isoformat(),
                "flair": post.get("link_flair_text"),
                "domain": post.get("domain"),
                "is_self": post.get("is_self", False),
                "keywords": self._extract_keywords_from_post(title, selftext, subreddit),
                "post_type": "self_post" if post.get("is_self") else "link_post"
            }
            
            return CollectionResult(
                source_id=f"reddit_post_{post_id}",
                source_type=self.source_type,
                collected_at=datetime.now(),
                data=post_data,
                metadata={
                    "source": "reddit",
                    "subreddit": subreddit,
                    "content_type": "post",
                    "score": post.get("score", 0)
                },
                raw_content=json.dumps(post, indent=2)
            )
            
        except Exception as e:
            print(f"Error parsing Reddit post: {e}")
            return None
    
    async def _parse_reddit_comment(self, 
                                  comment: Dict[str, Any], 
                                  subreddit: str,
                                  post_id: str,
                                  post_title: str) -> Optional[CollectionResult]:
        """Parse a Reddit comment into a CollectionResult."""
        
        try:
            comment_id = comment.get("id")
            body = comment.get("body", "")
            
            # Skip deleted/removed comments
            if body in ["[deleted]", "[removed]"] or not body.strip():
                return None
            
            # Extract metadata
            created_utc = comment.get("created_utc", 0)
            created_date = datetime.fromtimestamp(created_utc) if created_utc else datetime.now()
            
            comment_data = {
                "id": comment_id,
                "content": body,
                "post_id": post_id,
                "post_title": post_title,
                "subreddit": subreddit,
                "author": comment.get("author", "[deleted]"),
                "score": comment.get("score", 0),
                "created_utc": created_utc,
                "created_date": created_date.isoformat(),
                "parent_id": comment.get("parent_id", ""),
                "permalink": f"{self.base_url}{comment.get('permalink', '')}",
                "keywords": self._extract_keywords_from_text(body),
                "content_type": "comment"
            }
            
            return CollectionResult(
                source_id=f"reddit_comment_{comment_id}",
                source_type=self.source_type,
                collected_at=datetime.now(),
                data=comment_data,
                metadata={
                    "source": "reddit",
                    "subreddit": subreddit,
                    "content_type": "comment",
                    "post_id": post_id,
                    "score": comment.get("score", 0)
                },
                raw_content=json.dumps(comment, indent=2)
            )
            
        except Exception as e:
            print(f"Error parsing Reddit comment: {e}")
            return None
    
    def _extract_keywords_from_post(self, title: str, selftext: str, subreddit: str) -> List[str]:
        """Extract keywords from Reddit post content."""
        keywords = [subreddit.lower()]
        
        # Simple keyword extraction from title and content
        text = f"{title} {selftext}".lower()
        
        # Common technical terms that might be relevant
        tech_keywords = [
            "ai", "ml", "machine learning", "deep learning", "neural network",
            "algorithm", "data science", "python", "tensorflow", "pytorch",
            "research", "paper", "study", "analysis", "model", "dataset",
            "blockchain", "crypto", "quantum", "robotics", "automation"
        ]
        
        for keyword in tech_keywords:
            if keyword in text:
                keywords.append(keyword)
        
        return list(set(keywords))
    
    def _extract_keywords_from_text(self, text: str) -> List[str]:
        """Extract keywords from text content."""
        keywords = []
        text_lower = text.lower()
        
        # Simple keyword extraction
        tech_keywords = [
            "ai", "ml", "machine learning", "deep learning", "neural network",
            "algorithm", "data science", "python", "tensorflow", "pytorch",
            "research", "paper", "study", "analysis", "model", "dataset"
        ]
        
        for keyword in tech_keywords:
            if keyword in text_lower:
                keywords.append(keyword)
        
        return keywords
    
    async def health_check(self) -> bool:
        """Check if Reddit API is accessible."""
        try:
            async with aiohttp.ClientSession(headers=self.headers) as session:
                # Test with a simple request to a reliable subreddit
                url = f"{self.base_url}/r/test.json"
                params = {"limit": 1}
                async with session.get(url, params=params) as response:
                    return response.status == 200
        except Exception:
            return False
    
    async def get_trending_subreddits(self) -> List[str]:
        """Get trending subreddits related to technology and research."""
        return [
            "MachineLearning", "science", "technology", "programming",
            "artificial", "deeplearning", "research", "datascience",
            "compsci", "AskScience", "artificial_intelligence", "singularity",
            "coding", "Python", "analytics", "statistics", "algorithms",
            "robotics", "quantum", "bioinformatics", "neuroscience"
        ]
    
    async def search_posts(self, 
                          query: str, 
                          subreddit: Optional[str] = None,
                          time_filter: str = "week") -> AsyncIterator[CollectionResult]:
        """Search for specific posts across Reddit."""
        
        async with aiohttp.ClientSession(headers=self.headers) as session:
            # Build search URL
            if subreddit:
                url = f"{self.base_url}/r/{subreddit}/search.json"
            else:
                url = f"{self.base_url}/search.json"
            
            params = {
                "q": query,
                "sort": "relevance",
                "t": time_filter,
                "limit": self.max_posts,
                "type": "link"
            }
            
            if subreddit:
                params["restrict_sr"] = "true"
            
            try:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        for post_data in data.get("data", {}).get("children", []):
                            post = post_data.get("data", {})
                            if post:
                                result = await self._parse_reddit_post(
                                    post, post.get("subreddit", "unknown")
                                )
                                if result:
                                    yield result
                                    
            except Exception as e:
                yield CollectionResult(
                    source_id=f"reddit_search_error",
                    source_type=self.source_type,
                    collected_at=datetime.now(),
                    data={},
                    metadata={"error": str(e), "query": query},
                    success=False,
                    error_message=f"Failed to search Reddit: {e}"
                )