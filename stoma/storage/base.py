"""Base classes for data storage."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, AsyncIterator
from dataclasses import asdict

from ..normalizers.base import NormalizedData
from ..collectors.base import SourceType


class BaseStorage(ABC):
    """Abstract base class for data storage backends."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to storage backend."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to storage backend."""
        pass
    
    @abstractmethod
    async def store(self, data: NormalizedData) -> str:
        """Store normalized data and return unique storage ID."""
        pass
    
    @abstractmethod
    async def retrieve(self, storage_id: str) -> Optional[NormalizedData]:
        """Retrieve data by storage ID."""
        pass
    
    @abstractmethod
    async def search(
        self, 
        query: str,
        source_types: Optional[List[SourceType]] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        limit: int = 100
    ) -> AsyncIterator[NormalizedData]:
        """Search stored data."""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if storage backend is healthy."""
        pass


class PostgreSQLStorage(BaseStorage):
    """PostgreSQL storage implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Handle both connection_string format and individual fields
        if "connection_string" in config:
            self.connection_string = config["connection_string"]
        else:
            # Build connection string from individual fields
            database = config.get("database", "stoma")
            user = config.get("user", "postgres")
            host = config.get("host")
            port = config.get("port", 5432)
            password = config.get("password")
            
            if host:
                # TCP connection
                if password:
                    self.connection_string = f"postgresql://{user}:{password}@{host}:{port}/{database}"
                else:
                    self.connection_string = f"postgresql://{user}@{host}:{port}/{database}"
            else:
                # Unix socket connection (peer authentication)
                self.connection_string = f"postgresql://{user}@/{database}"
        
        self.pool = None
    
    async def connect(self) -> None:
        """Connect to PostgreSQL."""
        import asyncpg
        
        # For Unix socket connections, use explicit parameters instead of connection string
        if "host" not in self.config:
            # Use Unix socket connection with explicit parameters
            self.pool = await asyncpg.create_pool(
                database=self.config.get("database", "stoma"),
                user=self.config.get("user", "postgres"),
                port=self.config.get("port", 5432)
                # No host parameter = Unix socket
            )
        else:
            # Use connection string for TCP connections
            self.pool = await asyncpg.create_pool(self.connection_string)
            
        await self._create_tables()
    
    async def ensure_tables(self) -> None:
        """Ensure database tables exist (for setup scripts)."""
        await self._create_tables()
    
    async def disconnect(self) -> None:
        """Disconnect from PostgreSQL."""
        if self.pool:
            await self.pool.close()
    
    async def _create_tables(self) -> None:
        """Create necessary tables if they don't exist."""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS normalized_data (
            id TEXT PRIMARY KEY,
            source_type TEXT NOT NULL,
            source_id TEXT NOT NULL,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            summary TEXT,
            authors JSONB,
            published_date TIMESTAMP,
            collected_date TIMESTAMP NOT NULL,
            url TEXT,
            keywords JSONB,
            categories JSONB,
            tags JSONB,
            metrics JSONB,
            raw_data JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_normalized_data_source_type 
        ON normalized_data(source_type);
        
        CREATE INDEX IF NOT EXISTS idx_normalized_data_published_date 
        ON normalized_data(published_date);
        
        CREATE INDEX IF NOT EXISTS idx_normalized_data_collected_date 
        ON normalized_data(collected_date);
        
        CREATE INDEX IF NOT EXISTS idx_normalized_data_title_content 
        ON normalized_data USING gin(to_tsvector('english', title || ' ' || content));
        """
        
        async with self.pool.acquire() as conn:
            await conn.execute(create_table_sql)
    
    async def store(self, data: NormalizedData) -> str:
        """Store data in PostgreSQL."""
        insert_sql = """
        INSERT INTO normalized_data (
            id, source_type, source_id, title, content, summary,
            authors, published_date, collected_date, url,
            keywords, categories, tags, metrics, raw_data
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
        ON CONFLICT (id) DO UPDATE SET
            title = EXCLUDED.title,
            content = EXCLUDED.content,
            summary = EXCLUDED.summary,
            authors = EXCLUDED.authors,
            published_date = EXCLUDED.published_date,
            url = EXCLUDED.url,
            keywords = EXCLUDED.keywords,
            categories = EXCLUDED.categories,
            tags = EXCLUDED.tags,
            metrics = EXCLUDED.metrics,
            raw_data = EXCLUDED.raw_data,
            updated_at = CURRENT_TIMESTAMP
        """
        
        import json
        
        async with self.pool.acquire() as conn:
            await conn.execute(
                insert_sql,
                data.id,
                data.source_type.value,
                data.source_id,
                data.title,
                data.content,
                data.summary,
                json.dumps(data.authors),
                data.published_date.replace(tzinfo=None) if data.published_date else None,
                data.collected_date.replace(tzinfo=None) if data.collected_date else None,
                data.url,
                json.dumps(data.keywords),
                json.dumps(data.categories),
                json.dumps(data.tags),
                json.dumps(data.metrics),
                json.dumps(data.raw_data)
            )
        
        return data.id
    
    async def retrieve(self, storage_id: str) -> Optional[NormalizedData]:
        """Retrieve data by ID."""
        select_sql = "SELECT * FROM normalized_data WHERE id = $1"
        
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(select_sql, storage_id)
            if row:
                return self._row_to_normalized_data(row)
            return None
    
    async def search(
        self,
        query: str,
        source_types: Optional[List[SourceType]] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        limit: int = 100
    ) -> AsyncIterator[NormalizedData]:
        """Search data with full-text search."""
        conditions = ["to_tsvector('english', title || ' ' || content) @@ plainto_tsquery($1)"]
        params = [query]
        param_count = 1
        
        if source_types:
            param_count += 1
            conditions.append(f"source_type = ANY(${param_count})")
            params.append([st.value for st in source_types])
        
        if date_from:
            param_count += 1
            conditions.append(f"published_date >= ${param_count}")
            params.append(date_from)
        
        if date_to:
            param_count += 1
            conditions.append(f"published_date <= ${param_count}")
            params.append(date_to)
        
        param_count += 1
        search_sql = f"""
        SELECT * FROM normalized_data 
        WHERE {' AND '.join(conditions)}
        ORDER BY published_date DESC
        LIMIT ${param_count}
        """
        params.append(limit)
        
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                async for row in conn.cursor(search_sql, *params):
                    yield self._row_to_normalized_data(row)
    
    async def health_check(self) -> bool:
        """Check PostgreSQL connection."""
        try:
            async with self.pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            return True
        except Exception:
            return False
    
    def _row_to_normalized_data(self, row) -> NormalizedData:
        """Convert database row to NormalizedData."""
        import json
        
        return NormalizedData(
            id=row["id"],
            source_type=SourceType(row["source_type"]),
            source_id=row["source_id"],
            title=row["title"],
            content=row["content"],
            summary=row["summary"],
            authors=json.loads(row["authors"]) if row["authors"] else [],
            published_date=row["published_date"],
            collected_date=row["collected_date"],
            url=row["url"],
            keywords=json.loads(row["keywords"]) if row["keywords"] else [],
            categories=json.loads(row["categories"]) if row["categories"] else [],
            tags=json.loads(row["tags"]) if row["tags"] else [],
            metrics=json.loads(row["metrics"]) if row["metrics"] else {},
            raw_data=json.loads(row["raw_data"]) if row["raw_data"] else {}
        )