#!/usr/bin/env python3
"""
view_processing_log.py

Interactive viewer for processing_log.csv
Shows latest processing state for each document with metrics and timing.

Usage:
    python view_processing_log.py
    python view_processing_log.py --log-file custom_log.csv
    python view_processing_log.py --format table  # or json
    python view_processing_log.py --doc-id "FY25 Q1"  # filter by doc
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class ProcessingLogViewer:
    """Viewer for processing log CSV"""

    def __init__(self, log_file: str = "processing_log.csv"):
        self.log_file = Path(log_file)
        self.logs = []
        self.docs = {}

    def load_logs(self) -> bool:
        """Load and parse log file"""
        if not self.log_file.exists():
            print(f"❌ Log file not found: {self.log_file}")
            print(f"   Run the pipeline first to generate logs")
            return False

        try:
            with open(self.log_file, 'r', newline='') as f:
                reader = csv.DictReader(f)
                self.logs = list(reader)

            if not self.logs:
                print("⚠️  Log file is empty")
                return False

            print(f"✅ Loaded {len(self.logs)} log entries")
            return True

        except Exception as e:
            print(f"❌ Error loading log file: {e}")
            return False

    def aggregate_by_document(self) -> Dict[str, Dict]:
        """Aggregate logs by document ID to get latest state"""
        self.docs = {}

        for log in self.logs:
            doc_id = log['doc_id']
            stage = log['stage']

            if doc_id not in self.docs:
                self.docs[doc_id] = {
                    'doc_id': doc_id,
                    'stages_completed': [],
                    'latest_timestamp': None,
                    'latest_batch_uuid': None,
                    'pages': None,
                    'images_extracted': None,
                    'images_sent': None,
                    'images_downloaded': None,
                    'images_filtered_in': None,
                    'images_filtered_out': None,
                    'images_integrated': None,
                    'total_duration': 0.0,
                    'conversion_duration': None,
                    'extraction_duration': None,
                    'batch_duration': None,
                    'download_duration': None,
                    'filter_duration': None,
                    'integration_duration': None,
                    'status': 'success',
                }

            doc = self.docs[doc_id]

            # Track stages
            if stage not in doc['stages_completed']:
                doc['stages_completed'].append(stage)

            # Update timestamp
            timestamp = log['timestamp']
            if not doc['latest_timestamp'] or timestamp > doc['latest_timestamp']:
                doc['latest_timestamp'] = timestamp

            # Update batch UUID
            if log['batch_uuid']:
                doc['latest_batch_uuid'] = log['batch_uuid']

            # Update metrics based on stage
            if log['pages']:
                doc['pages'] = int(log['pages'])

            if log['images_extracted']:
                doc['images_extracted'] = int(log['images_extracted'])

            if log['images_sent_to_batch']:
                doc['images_sent'] = int(log['images_sent_to_batch'])

            if log['images_downloaded']:
                doc['images_downloaded'] = int(log['images_downloaded'])

            if log['images_filtered_in']:
                doc['images_filtered_in'] = int(log['images_filtered_in'])

            if log['images_filtered_out']:
                doc['images_filtered_out'] = int(log['images_filtered_out'])

            if log['images_integrated']:
                doc['images_integrated'] = int(log['images_integrated'])

            # Track duration by stage
            if log['duration_seconds']:
                duration = float(log['duration_seconds'])
                doc['total_duration'] += duration

                if stage == 'conversion':
                    doc['conversion_duration'] = duration
                elif stage == 'extraction':
                    doc['extraction_duration'] = duration
                elif stage == 'batch_creation':
                    doc['batch_duration'] = duration
                elif stage == 'download':
                    doc['download_duration'] = duration
                elif stage == 'filter':
                    doc['filter_duration'] = duration
                elif stage == 'integration':
                    doc['integration_duration'] = duration

            # Track status
            if log['status'] == 'failed':
                doc['status'] = 'failed'

        return self.docs

    def get_completion_percentage(self, doc: Dict) -> float:
        """Calculate completion percentage for a document"""
        all_stages = ['conversion', 'extraction', 'batch_creation', 'download', 'filter', 'integration']
        completed = len([s for s in doc['stages_completed'] if s in all_stages])
        return (completed / len(all_stages)) * 100

    def get_stage_emoji(self, doc: Dict, stage: str) -> str:
        """Get emoji indicator for stage"""
        if stage in doc['stages_completed']:
            return "✅"
        return "⏸️ "

    def print_table_view(self, filter_doc_id: Optional[str] = None):
        """Print documents in table format"""
        docs_list = list(self.docs.values())

        # Filter if requested
        if filter_doc_id:
            docs_list = [d for d in docs_list if filter_doc_id.lower() in d['doc_id'].lower()]
            if not docs_list:
                print(f"❌ No documents found matching '{filter_doc_id}'")
                return

        # Sort by latest timestamp
        docs_list.sort(key=lambda x: x['latest_timestamp'] or '', reverse=True)

        print("\n" + "=" * 150)
        print("📊 DOCUMENT PROCESSING STATUS")
        print("=" * 150)
        print(f"Total Documents: {len(docs_list)}")
        print("=" * 150)

        for doc in docs_list:
            completion = self.get_completion_percentage(doc)
            status_icon = "✅" if completion == 100 else "🔄" if completion > 0 else "⏸️ "

            print(f"\n{status_icon} {doc['doc_id']}")
            print(f"   Completion: {completion:.0f}% | Batch UUID: {doc['latest_batch_uuid'] or 'N/A'}")

            # Pipeline stages
            stages_str = (
                f"{self.get_stage_emoji(doc, 'conversion')} Convert → "
                f"{self.get_stage_emoji(doc, 'extraction')} Extract → "
                f"{self.get_stage_emoji(doc, 'batch_creation')} Batch → "
                f"{self.get_stage_emoji(doc, 'download')} Download → "
                f"{self.get_stage_emoji(doc, 'filter')} Filter → "
                f"{self.get_stage_emoji(doc, 'integration')} Integrate"
            )
            print(f"   Pipeline: {stages_str}")

            # Metrics
            metrics_parts = []
            if doc['pages']:
                metrics_parts.append(f"{doc['pages']} pages")
            if doc['images_extracted']:
                metrics_parts.append(f"{doc['images_extracted']} images")
            if doc['images_filtered_in'] is not None:
                metrics_parts.append(f"{doc['images_filtered_in']} filtered")
            if doc['images_integrated']:
                metrics_parts.append(f"{doc['images_integrated']} integrated")

            if metrics_parts:
                print(f"   Metrics:  {' | '.join(metrics_parts)}")

            # Timing
            timing_parts = []
            if doc['conversion_duration']:
                timing_parts.append(f"Convert: {doc['conversion_duration']:.1f}s")
            if doc['extraction_duration']:
                timing_parts.append(f"Extract: {doc['extraction_duration']:.1f}s")
            if doc['download_duration']:
                timing_parts.append(f"Download: {doc['download_duration']:.1f}s")
            if doc['filter_duration']:
                timing_parts.append(f"Filter: {doc['filter_duration']:.1f}s")

            if timing_parts:
                print(f"   Timing:   {' | '.join(timing_parts)}")

            if doc['total_duration'] > 0:
                print(f"   Total:    {doc['total_duration']:.1f}s ({doc['total_duration']/60:.1f} min)")

            # Latest activity
            if doc['latest_timestamp']:
                try:
                    dt = datetime.fromisoformat(doc['latest_timestamp'])
                    time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                    print(f"   Updated:  {time_str}")
                except:
                    print(f"   Updated:  {doc['latest_timestamp']}")

        print("\n" + "=" * 150)

    def print_json_view(self, filter_doc_id: Optional[str] = None):
        """Print documents in JSON format"""
        docs_list = list(self.docs.values())

        # Filter if requested
        if filter_doc_id:
            docs_list = [d for d in docs_list if filter_doc_id.lower() in d['doc_id'].lower()]

        # Add completion percentage
        for doc in docs_list:
            doc['completion_percentage'] = self.get_completion_percentage(doc)

        print(json.dumps(docs_list, indent=2))

    def print_summary_stats(self):
        """Print overall summary statistics"""
        if not self.docs:
            return

        total_docs = len(self.docs)
        completed_docs = len([d for d in self.docs.values() if self.get_completion_percentage(d) == 100])
        in_progress_docs = len([d for d in self.docs.values() if 0 < self.get_completion_percentage(d) < 100])
        not_started_docs = len([d for d in self.docs.values() if self.get_completion_percentage(d) == 0])

        total_pages = sum(d['pages'] or 0 for d in self.docs.values())
        total_images_extracted = sum(d['images_extracted'] or 0 for d in self.docs.values())
        total_images_filtered = sum(d['images_filtered_in'] or 0 for d in self.docs.values())
        total_images_integrated = sum(d['images_integrated'] or 0 for d in self.docs.values())

        total_time = sum(d['total_duration'] for d in self.docs.values())

        print("\n" + "=" * 80)
        print("📈 SUMMARY STATISTICS")
        print("=" * 80)
        print(f"Documents:        {total_docs} total")
        print(f"  ✅ Completed:   {completed_docs}")
        print(f"  🔄 In Progress: {in_progress_docs}")
        print(f"  ⏸️  Not Started: {not_started_docs}")
        print()
        print(f"Pages Processed:  {total_pages}")
        print(f"Images Extracted: {total_images_extracted}")
        print(f"Images Filtered:  {total_images_filtered}")
        print(f"Images Integrated: {total_images_integrated}")
        print()
        print(f"Total Time:       {total_time:.1f}s ({total_time/60:.1f} min)")
        if completed_docs > 0:
            avg_time = total_time / completed_docs
            print(f"Avg Time/Doc:     {avg_time:.1f}s ({avg_time/60:.1f} min)")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="View processing log with document status and metrics"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="processing_log.csv",
        help="Path to log file (default: processing_log.csv)",
    )
    parser.add_argument(
        "--format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )
    parser.add_argument(
        "--doc-id",
        type=str,
        help="Filter by document ID (partial match)",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Show summary statistics",
    )

    args = parser.parse_args()

    # Initialize viewer
    viewer = ProcessingLogViewer(args.log_file)

    # Load logs
    if not viewer.load_logs():
        return 1

    # Aggregate by document
    viewer.aggregate_by_document()

    print()  # Blank line

    # Display based on format
    if args.format == "json":
        viewer.print_json_view(args.doc_id)
    else:
        viewer.print_table_view(args.doc_id)

    # Show summary if requested
    if args.summary:
        viewer.print_summary_stats()

    return 0


if __name__ == "__main__":
    sys.exit(main())
