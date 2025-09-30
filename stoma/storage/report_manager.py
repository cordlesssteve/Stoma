"""
Report Storage Manager for KnowHunt LLM Analysis Reports

Handles organized storage, indexing, and retrieval of LLM analysis reports.
"""

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import hashlib
import logging

from .database import DatabaseStorage

logger = logging.getLogger(__name__)


class ReportStorageManager:
    """Manages organized storage and indexing of LLM analysis reports."""

    def __init__(self, base_path: Optional[str] = None, use_postgresql: bool = True):
        """Initialize the report storage manager."""
        if base_path is None:
            # Default to project reports directory
            project_root = Path(__file__).parent.parent.parent
            base_path = project_root / "reports" / "llm_analysis"

        self.base_path = Path(base_path)
        self.index_db_path = self.base_path / "indexed" / "report_index.db"

        # Try to initialize PostgreSQL storage if requested
        self.db_storage = None
        if use_postgresql:
            try:
                self.db_storage = DatabaseStorage()
                if self.db_storage.test_connection():
                    # Create LLM analysis tables
                    self.db_storage.create_llm_analysis_tables()
                    logger.info("PostgreSQL storage available for LLM reports")
                else:
                    logger.warning("PostgreSQL connection failed, using SQLite fallback")
                    self.db_storage = None
            except Exception as e:
                logger.warning(f"PostgreSQL initialization failed, using SQLite fallback: {e}")
                self.db_storage = None

        # Ensure directories exist
        self._create_directory_structure()

        # Initialize SQLite as fallback if PostgreSQL not available
        if not self.db_storage:
            self._initialize_index_db()

    def _create_directory_structure(self):
        """Create the organized directory structure."""
        directories = [
            self.base_path / "raw",
            self.base_path / "processed",
            self.base_path / "indexed",
            self.base_path / "by_date",
            self.base_path / "by_provider" / "openai",
            self.base_path / "by_provider" / "anthropic",
            self.base_path / "by_provider" / "ollama"
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def _initialize_index_db(self):
        """Initialize the SQLite index database."""
        self.index_db_path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.index_db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id TEXT UNIQUE,
                    filename TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    research_quality_score REAL,
                    input_text_hash TEXT,
                    file_path TEXT NOT NULL,
                    novel_contributions_count INTEGER,
                    technical_innovations_count INTEGER,
                    business_implications_count INTEGER,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS report_keywords (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    report_id INTEGER,
                    keyword TEXT NOT NULL,
                    FOREIGN KEY (report_id) REFERENCES reports (id)
                )
            """)

            conn.execute("CREATE INDEX IF NOT EXISTS idx_document_id ON reports(document_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_provider ON reports(provider)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON reports(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_quality_score ON reports(research_quality_score)")

    def generate_report_path(self, provider: str, model: str, document_id: Optional[str] = None,
                           timestamp: Optional[datetime] = None) -> Path:
        """Generate an organized path for storing a report."""
        if timestamp is None:
            timestamp = datetime.now()

        if document_id is None:
            document_id = f"analysis_{timestamp.strftime('%Y%m%d_%H%M%S')}"

        # Create date-based path
        date_str = timestamp.strftime('%Y-%m')
        date_path = self.base_path / "by_date" / date_str
        date_path.mkdir(parents=True, exist_ok=True)

        # Create provider-based path
        provider_path = self.base_path / "by_provider" / provider.lower()
        provider_path.mkdir(parents=True, exist_ok=True)

        # Generate filename with model info
        safe_model = model.replace(":", "_").replace("/", "_")
        filename = f"{document_id}_{provider}_{safe_model}_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"

        # Use provider path as primary storage
        return provider_path / filename

    def save_analysis_report(self, report_data: Dict[str, Any],
                           auto_path: bool = True,
                           custom_path: Optional[str] = None) -> Path:
        """Save an analysis report with automatic organization and indexing."""

        if auto_path:
            # Generate organized path automatically
            timestamp = datetime.fromisoformat(report_data.get("timestamp", datetime.now().isoformat()))
            provider = report_data.get("provider", "unknown")
            model = report_data.get("model", "unknown")
            document_id = report_data.get("analysis", {}).get("document_id")

            file_path = self.generate_report_path(provider, model, document_id, timestamp)
        else:
            if custom_path is None:
                raise ValueError("custom_path must be provided when auto_path=False")
            file_path = Path(custom_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)

        # Save the report
        with open(file_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)

        # Add file path to report data for database storage
        report_data['_file_path'] = str(file_path)

        # Index the report (PostgreSQL or SQLite)
        if self.db_storage:
            # Use PostgreSQL storage
            self.db_storage.store_llm_analysis(report_data)
        else:
            # Use SQLite fallback
            self._index_report(report_data, file_path)

        return file_path

    def _index_report(self, report_data: Dict[str, Any], file_path: Path):
        """Add report metadata to the search index."""
        analysis = report_data.get("analysis", {})

        # Generate input text hash for deduplication
        input_text = report_data.get("input_text", "")
        input_hash = hashlib.sha256(input_text.encode()).hexdigest()[:16]

        with sqlite3.connect(self.index_db_path) as conn:
            # Insert or update main report record
            cursor = conn.execute("""
                INSERT OR REPLACE INTO reports
                (document_id, filename, provider, model, timestamp, research_quality_score,
                 input_text_hash, file_path, novel_contributions_count,
                 technical_innovations_count, business_implications_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                analysis.get("document_id", ""),
                file_path.name,
                report_data.get("provider", ""),
                report_data.get("model", ""),
                report_data.get("timestamp", ""),
                analysis.get("research_quality_score", 0),
                input_hash,
                str(file_path),
                len(analysis.get("novel_contributions", [])),
                len(analysis.get("technical_innovations", [])),
                len(analysis.get("business_implications", []))
            ))

            report_id = cursor.lastrowid

            # Index keywords
            keywords = analysis.get("concept_keywords", [])
            if keywords:
                # Clear existing keywords for this report
                conn.execute("DELETE FROM report_keywords WHERE report_id = ?", (report_id,))

                # Insert new keywords
                for keyword in keywords:
                    if isinstance(keyword, str) and keyword.strip():
                        conn.execute(
                            "INSERT INTO report_keywords (report_id, keyword) VALUES (?, ?)",
                            (report_id, keyword.strip().lower())
                        )

    def search_reports(self, query: Optional[str] = None, provider: Optional[str] = None,
                      min_quality_score: Optional[float] = None,
                      limit: int = 50) -> List[Dict[str, Any]]:
        """Search for reports based on various criteria."""

        if self.db_storage:
            # Use PostgreSQL search
            results = self.db_storage.search_llm_analysis(
                query=query,
                provider=provider,
                min_quality_score=min_quality_score,
                limit=limit
            )

            # Convert PostgreSQL results to match expected format
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "document_id": result.get("document_id", ""),
                    "filename": Path(result.get("file_path", "")).name if result.get("file_path") else "",
                    "provider": result.get("provider", ""),
                    "model": result.get("model", ""),
                    "timestamp": str(result.get("timestamp", "")),
                    "research_quality_score": result.get("research_quality_score", 0),
                    "file_path": result.get("file_path", ""),
                    "novel_contributions_count": len(result.get("novel_contributions", [])),
                    "technical_innovations_count": len(result.get("technical_innovations", [])),
                    "business_implications_count": len(result.get("business_implications", [])),
                    "keywords": ", ".join(result.get("concept_keywords", [])[:5])
                })

            return formatted_results

        else:
            # Use SQLite fallback
            sql = """
                SELECT r.*, GROUP_CONCAT(k.keyword, ', ') as keywords
                FROM reports r
                LEFT JOIN report_keywords k ON r.id = k.report_id
                WHERE 1=1
            """
            params = []

            if query:
                sql += " AND (r.document_id LIKE ? OR k.keyword LIKE ?)"
                params.extend([f"%{query}%", f"%{query}%"])

            if provider:
                sql += " AND r.provider = ?"
                params.append(provider)

            if min_quality_score is not None:
                sql += " AND r.research_quality_score >= ?"
                params.append(min_quality_score)

            sql += " GROUP BY r.id ORDER BY r.timestamp DESC LIMIT ?"
            params.append(limit)

            with sqlite3.connect(self.index_db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(sql, params)

                results = []
                for row in cursor.fetchall():
                    results.append({
                        "document_id": row["document_id"],
                        "filename": row["filename"],
                        "provider": row["provider"],
                        "model": row["model"],
                        "timestamp": row["timestamp"],
                        "research_quality_score": row["research_quality_score"],
                        "file_path": row["file_path"],
                        "novel_contributions_count": row["novel_contributions_count"],
                        "technical_innovations_count": row["technical_innovations_count"],
                        "business_implications_count": row["business_implications_count"],
                        "keywords": row["keywords"] if row["keywords"] else ""
                    })

                return results

    def get_report_by_id(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific report by document ID."""

        if self.db_storage:
            # Use PostgreSQL storage
            db_result = self.db_storage.get_llm_analysis_by_id(document_id)
            if db_result:
                # Try to load full report from file if available
                file_path = db_result.get('file_path')
                if file_path and Path(file_path).exists():
                    try:
                        with open(file_path, 'r') as f:
                            full_report = json.load(f)

                        # Add database metadata
                        full_report["_database_metadata"] = db_result
                        return full_report
                    except Exception:
                        # Return database data if file loading fails
                        return db_result
                else:
                    return db_result

        else:
            # Use SQLite fallback
            with sqlite3.connect(self.index_db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    "SELECT * FROM reports WHERE document_id = ?",
                    (document_id,)
                )
                row = cursor.fetchone()

                if row:
                    # Load the full report content
                    file_path = Path(row["file_path"])
                    if file_path.exists():
                        with open(file_path, 'r') as f:
                            full_report = json.load(f)

                        # Add index metadata
                        full_report["_index_metadata"] = dict(row)
                        return full_report

        return None

    def get_storage_statistics(self) -> Dict[str, Any]:
        """Get statistics about stored reports."""

        if self.db_storage:
            # Use PostgreSQL storage statistics
            stats = self.db_storage.get_llm_analysis_stats()

            # Add disk usage calculation
            total_size = 0
            for provider_dir in (self.base_path / "by_provider").iterdir():
                if provider_dir.is_dir():
                    for report_file in provider_dir.glob("*.json"):
                        total_size += report_file.stat().st_size

            stats["disk_usage_mb"] = round(total_size / (1024 * 1024), 2)

            return stats

        else:
            # Use SQLite fallback
            with sqlite3.connect(self.index_db_path) as conn:
                conn.row_factory = sqlite3.Row

                stats = {}

                # Total reports
                cursor = conn.execute("SELECT COUNT(*) as total FROM reports")
                stats["total_reports"] = cursor.fetchone()["total"]

                # By provider
                cursor = conn.execute("""
                    SELECT provider, COUNT(*) as count, AVG(research_quality_score) as avg_score
                    FROM reports
                    GROUP BY provider
                    ORDER BY count DESC
                """)
                stats["by_provider"] = [dict(row) for row in cursor.fetchall()]

                # Quality score distribution
                cursor = conn.execute("""
                    SELECT
                        COUNT(CASE WHEN research_quality_score >= 8 THEN 1 END) as high_quality,
                        COUNT(CASE WHEN research_quality_score >= 5 AND research_quality_score < 8 THEN 1 END) as medium_quality,
                        COUNT(CASE WHEN research_quality_score < 5 THEN 1 END) as low_quality,
                        AVG(research_quality_score) as avg_quality_score
                    FROM reports
                """)
                quality_stats = dict(cursor.fetchone())
                stats["quality_distribution"] = quality_stats

                # Recent activity (last 30 days)
                cursor = conn.execute("""
                    SELECT COUNT(*) as recent_reports
                    FROM reports
                    WHERE timestamp >= datetime('now', '-30 days')
                """)
                stats["recent_activity"] = cursor.fetchone()["recent_reports"]

                # Disk usage
                total_size = 0
                for provider_dir in (self.base_path / "by_provider").iterdir():
                    if provider_dir.is_dir():
                        for report_file in provider_dir.glob("*.json"):
                            total_size += report_file.stat().st_size

                stats["disk_usage_mb"] = round(total_size / (1024 * 1024), 2)

                return stats