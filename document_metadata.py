"""
Document Metadata Manager

Tracks downloaded documents, conversion status, and metadata using SQLite.
"""

import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from contextlib import contextmanager


class MetadataManager:
    """Manages document metadata using SQLite database."""

    def __init__(self, db_path: str = "to_process/metadata.db"):
        """
        Initialize metadata manager.

        Args:
            db_path: Path to SQLite database file.
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    def _init_database(self):
        """Initialize database schema."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    document_id INTEGER PRIMARY KEY,
                    company_id INTEGER,
                    ticker TEXT,
                    company_name TEXT,
                    country TEXT,
                    document_type TEXT,
                    filing_date TEXT,
                    title TEXT,
                    pdf_filename TEXT,
                    md_filename TEXT,
                    download_status TEXT DEFAULT 'pending',
                    download_timestamp TEXT,
                    conversion_timestamp TEXT,
                    error_message TEXT,
                    pdf_url TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create indexes for faster queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_download_status
                ON documents(download_status)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_company_id
                ON documents(company_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_document_type
                ON documents(document_type)
            """)

            conn.commit()

    @contextmanager
    def _get_connection(self):
        """Get database connection context manager."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def add_document(
        self,
        document_id: int,
        company_id: int,
        ticker: str,
        company_name: str,
        country: str,
        document_type: str,
        filing_date: Optional[str],
        title: str,
        pdf_url: Optional[str] = None,
    ) -> bool:
        """
        Add a document to the metadata database.

        Args:
            document_id: Unique document identifier.
            company_id: Company identifier.
            ticker: Company ticker symbol.
            company_name: Company name.
            country: Company country.
            document_type: Type of document (filing, slides, etc.).
            filing_date: Document filing date.
            title: Document title.
            pdf_url: URL to download PDF (if available).

        Returns:
            True if document was added, False if it already exists.
        """
        with self._get_connection() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO documents (
                        document_id, company_id, ticker, company_name, country,
                        document_type, filing_date, title, pdf_url
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        document_id,
                        company_id,
                        ticker,
                        company_name,
                        country,
                        document_type,
                        filing_date,
                        title,
                        pdf_url,
                    ),
                )
                conn.commit()
                return True
            except sqlite3.IntegrityError:
                # Document already exists
                return False

    def mark_downloaded(self, document_id: int, pdf_filename: str):
        """Mark a document as successfully downloaded."""
        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE documents
                SET download_status = 'downloaded',
                    pdf_filename = ?,
                    download_timestamp = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE document_id = ?
                """,
                (pdf_filename, datetime.now().isoformat(), document_id),
            )
            conn.commit()

    def mark_converted(self, document_id: int, md_filename: str):
        """Mark a document as successfully converted."""
        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE documents
                SET download_status = 'converted',
                    md_filename = ?,
                    conversion_timestamp = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE document_id = ?
                """,
                (md_filename, datetime.now().isoformat(), document_id),
            )
            conn.commit()

    def mark_failed(self, document_id: int, error_message: str, status: str = "failed"):
        """Mark a document as failed with an error message."""
        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE documents
                SET download_status = ?,
                    error_message = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE document_id = ?
                """,
                (status, error_message, document_id),
            )
            conn.commit()

    def get_pending_downloads(self) -> List[Dict]:
        """Get all documents pending download."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM documents
                WHERE download_status = 'pending'
                ORDER BY document_id
                """
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_downloaded_documents(self) -> List[Dict]:
        """Get all documents that have been downloaded but not converted."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM documents
                WHERE download_status = 'downloaded'
                ORDER BY document_id
                """
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_document_by_id(self, document_id: int) -> Optional[Dict]:
        """Get document metadata by document ID."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM documents WHERE document_id = ?", (document_id,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def document_exists(self, document_id: int) -> bool:
        """Check if a document exists in the database."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT 1 FROM documents WHERE document_id = ? LIMIT 1", (document_id,)
            )
            return cursor.fetchone() is not None

    def get_statistics(self) -> Dict[str, int]:
        """Get statistics about documents in the database."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT
                    download_status,
                    COUNT(*) as count
                FROM documents
                GROUP BY download_status
                """
            )
            stats = {row["download_status"]: row["count"] for row in cursor.fetchall()}

            # Get total count
            cursor = conn.execute("SELECT COUNT(*) as total FROM documents")
            stats["total"] = cursor.fetchone()["total"]

            return stats

    def get_documents_by_status(self, status: str) -> List[Dict]:
        """Get all documents with a specific status."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM documents WHERE download_status = ? ORDER BY document_id",
                (status,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_failed_documents(self) -> List[Dict]:
        """Get all documents that failed to download or convert."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM documents
                WHERE download_status IN ('failed', 'download_failed', 'conversion_failed')
                ORDER BY document_id
                """
            )
            return [dict(row) for row in cursor.fetchall()]

    def reset_failed_documents(self):
        """Reset all failed documents back to pending status."""
        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE documents
                SET download_status = 'pending',
                    error_message = NULL,
                    updated_at = CURRENT_TIMESTAMP
                WHERE download_status IN ('failed', 'download_failed', 'conversion_failed')
                """
            )
            conn.commit()

    def cleanup_orphaned_entries(self, input_folder: Path):
        """Remove database entries for files that no longer exist on disk."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT document_id, pdf_filename FROM documents WHERE pdf_filename IS NOT NULL"
            )

            orphaned = []
            for row in cursor.fetchall():
                pdf_path = input_folder / row["pdf_filename"]
                if not pdf_path.exists():
                    orphaned.append(row["document_id"])

            if orphaned:
                placeholders = ",".join("?" * len(orphaned))
                conn.execute(
                    f"DELETE FROM documents WHERE document_id IN ({placeholders})",
                    orphaned,
                )
                conn.commit()

            return len(orphaned)


if __name__ == "__main__":
    # Example usage
    manager = MetadataManager()

    # Print statistics
    stats = manager.get_statistics()
    print("Document Statistics:")
    for status, count in stats.items():
        print(f"  {status}: {count}")
