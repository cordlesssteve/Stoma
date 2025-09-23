"""Synchronous database storage for NLP analysis."""

import logging
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import Dict, Any, Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class DatabaseStorage:
    """Synchronous PostgreSQL storage for NLP analysis."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize database storage.
        
        Args:
            config: Database configuration dict
        """
        if not config:
            # Default configuration for local development
            config = {
                "host": "localhost",
                "port": 5433,
                "database": "knowhunt",
                "user": "postgres"
            }
        
        self.config = config
        self._connection_params = self._build_connection_params()
        
    def _build_connection_params(self) -> Dict[str, Any]:
        """Build psycopg2 connection parameters."""
        params = {
            "database": self.config.get("database", "knowhunt"),
            "user": self.config.get("user", "postgres"),
            "port": self.config.get("port", 5433)
        }
        
        # Add host if specified (for TCP connections)
        if "host" in self.config:
            params["host"] = self.config["host"]
        
        # Add password if specified
        if "password" in self.config:
            params["password"] = self.config["password"]
        
        return params
    
    @contextmanager
    def get_connection(self):
        """Get a database connection context manager."""
        conn = None
        try:
            conn = psycopg2.connect(**self._connection_params)
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database connection error: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def test_connection(self) -> bool:
        """Test database connection."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    return True
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False