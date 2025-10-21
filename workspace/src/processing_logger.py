#!/usr/bin/env python3
"""
processing_logger.py

Centralized CSV logging for document processing pipeline.
Tracks processing metrics across all stages: extraction, batch creation,
downloading, filtering, and integration.

Usage:
    from processing_logger import ProcessingLogger

    logger = ProcessingLogger()
    logger.log_extraction(doc_id="my_doc", pages=10, images_extracted=25, duration_seconds=5.2)
"""

import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.pipeline.paths import LOGS_DIR


class ProcessingLogger:
    """CSV logger for document processing pipeline"""

    def __init__(self, log_file: str = str(LOGS_DIR / "processing_log.csv")):
        """
        Initialize the logger.

        Args:
            log_file: Path to CSV log file (default: processing_log.csv in current directory)
        """
        self.log_file = Path(log_file)
        self._ensure_log_exists()

    def _ensure_log_exists(self):
        """Create log file with headers if it doesn't exist"""
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.log_file.exists():
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp',
                    'doc_id',
                    'stage',
                    'batch_uuid',
                    'pages',
                    'images_extracted',
                    'images_sent_to_batch',
                    'images_downloaded',
                    'images_filtered_in',
                    'images_filtered_out',
                    'images_integrated',
                    'duration_seconds',
                    'status',
                    'notes'
                ])

    def log_conversion(
        self,
        doc_id: str,
        pages: int,
        duration_seconds: float,
        status: str = "success",
        notes: str = ""
    ):
        """
        Log PDF to Markdown conversion stage (Docling).

        Args:
            doc_id: Document identifier
            pages: Number of pages in document
            duration_seconds: Time taken for conversion
            status: Status (success/failed)
            notes: Additional notes
        """
        self._write_log(
            doc_id=doc_id,
            stage="conversion",
            batch_uuid="",
            pages=pages,
            duration_seconds=duration_seconds,
            status=status,
            notes=notes
        )

    def log_extraction(
        self,
        doc_id: str,
        pages: int,
        images_extracted: int,
        duration_seconds: float,
        status: str = "success",
        notes: str = ""
    ):
        """
        Log image extraction stage.

        Args:
            doc_id: Document identifier
            pages: Number of pages in document
            images_extracted: Number of images extracted
            duration_seconds: Time taken for extraction
            status: Status (success/failed)
            notes: Additional notes
        """
        self._write_log(
            doc_id=doc_id,
            stage="extraction",
            batch_uuid="",
            pages=pages,
            images_extracted=images_extracted,
            duration_seconds=duration_seconds,
            status=status,
            notes=notes
        )

    def log_batch_creation(
        self,
        doc_id: str,
        batch_uuid: str,
        images_sent: int,
        duration_seconds: float,
        status: str = "success",
        notes: str = ""
    ):
        """
        Log batch creation stage.

        Args:
            doc_id: Document identifier
            batch_uuid: UUID for this batch run
            images_sent: Number of images sent to batch
            duration_seconds: Time taken
            status: Status (success/failed)
            notes: Additional notes
        """
        self._write_log(
            doc_id=doc_id,
            stage="batch_creation",
            batch_uuid=batch_uuid,
            images_sent_to_batch=images_sent,
            duration_seconds=duration_seconds,
            status=status,
            notes=notes
        )

    def log_download(
        self,
        doc_id: str,
        batch_uuid: str,
        images_downloaded: int,
        duration_seconds: float,
        status: str = "success",
        notes: str = ""
    ):
        """
        Log batch download stage.

        Args:
            doc_id: Document identifier
            batch_uuid: UUID for this batch run
            images_downloaded: Number of descriptions downloaded
            duration_seconds: Time taken
            status: Status (success/failed)
            notes: Additional notes
        """
        self._write_log(
            doc_id=doc_id,
            stage="download",
            batch_uuid=batch_uuid,
            images_downloaded=images_downloaded,
            duration_seconds=duration_seconds,
            status=status,
            notes=notes
        )

    def log_filter(
        self,
        doc_id: str,
        batch_uuid: str,
        images_filtered_in: int,
        images_filtered_out: int,
        duration_seconds: float,
        status: str = "success",
        notes: str = ""
    ):
        """
        Log filtering stage.

        Args:
            doc_id: Document identifier
            batch_uuid: UUID for this batch run
            images_filtered_in: Number of images kept after filtering
            images_filtered_out: Number of images removed
            duration_seconds: Time taken
            status: Status (success/failed)
            notes: Additional notes
        """
        self._write_log(
            doc_id=doc_id,
            stage="filter",
            batch_uuid=batch_uuid,
            images_filtered_in=images_filtered_in,
            images_filtered_out=images_filtered_out,
            duration_seconds=duration_seconds,
            status=status,
            notes=notes
        )

    def log_integration(
        self,
        doc_id: str,
        batch_uuid: str,
        images_integrated: int,
        duration_seconds: float,
        status: str = "success",
        notes: str = ""
    ):
        """
        Log integration stage.

        Args:
            doc_id: Document identifier
            batch_uuid: UUID for this batch run
            images_integrated: Number of descriptions integrated into markdown
            duration_seconds: Time taken
            status: Status (success/failed)
            notes: Additional notes
        """
        self._write_log(
            doc_id=doc_id,
            stage="integration",
            batch_uuid=batch_uuid,
            images_integrated=images_integrated,
            duration_seconds=duration_seconds,
            status=status,
            notes=notes
        )

    def _write_log(
        self,
        doc_id: str,
        stage: str,
        batch_uuid: str = "",
        pages: Optional[int] = None,
        images_extracted: Optional[int] = None,
        images_sent_to_batch: Optional[int] = None,
        images_downloaded: Optional[int] = None,
        images_filtered_in: Optional[int] = None,
        images_filtered_out: Optional[int] = None,
        images_integrated: Optional[int] = None,
        duration_seconds: float = 0.0,
        status: str = "success",
        notes: str = ""
    ):
        """Write a log entry to CSV"""
        timestamp = datetime.now().isoformat()

        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                doc_id,
                stage,
                batch_uuid,
                pages or '',
                images_extracted or '',
                images_sent_to_batch or '',
                images_downloaded or '',
                images_filtered_in or '',
                images_filtered_out or '',
                images_integrated or '',
                f"{duration_seconds:.2f}",
                status,
                notes
            ])

    def get_document_summary(self, doc_id: str) -> dict:
        """
        Get a summary of processing stages for a document.

        Args:
            doc_id: Document identifier

        Returns:
            Dictionary with processing statistics
        """
        if not self.log_file.exists():
            return {}

        summary = {
            'doc_id': doc_id,
            'stages_completed': [],
            'total_duration': 0.0,
            'pages': None,
            'images_extracted': None,
            'images_in_latest_batch': None,
            'images_filtered_in': None,
            'images_integrated': None,
            'latest_batch_uuid': None
        }

        with open(self.log_file, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['doc_id'] == doc_id:
                    summary['stages_completed'].append(row['stage'])

                    if row['duration_seconds']:
                        summary['total_duration'] += float(row['duration_seconds'])

                    if row['pages']:
                        summary['pages'] = int(row['pages'])

                    if row['images_extracted']:
                        summary['images_extracted'] = int(row['images_extracted'])

                    if row['images_sent_to_batch']:
                        summary['images_in_latest_batch'] = int(row['images_sent_to_batch'])

                    if row['images_filtered_in']:
                        summary['images_filtered_in'] = int(row['images_filtered_in'])

                    if row['images_integrated']:
                        summary['images_integrated'] = int(row['images_integrated'])

                    if row['batch_uuid']:
                        summary['latest_batch_uuid'] = row['batch_uuid']

        return summary


def main():
    """Test the logger"""
    logger = ProcessingLogger("test_log.csv")

    # Test logging
    logger.log_extraction("test_doc", pages=10, images_extracted=25, duration_seconds=5.2)
    logger.log_batch_creation("test_doc", batch_uuid="abc123", images_sent=25, duration_seconds=1.5)
    logger.log_download("test_doc", batch_uuid="abc123", images_downloaded=25, duration_seconds=30.0)
    logger.log_filter("test_doc", batch_uuid="abc123", images_filtered_in=10, images_filtered_out=15, duration_seconds=45.0)
    logger.log_integration("test_doc", batch_uuid="abc123", images_integrated=10, duration_seconds=2.0)

    # Get summary
    summary = logger.get_document_summary("test_doc")
    print("\nDocument Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
