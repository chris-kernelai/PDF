#!/usr/bin/env python3
"""
Count files in important pipeline directories and analyze processing log.
"""
import os
import csv
from pathlib import Path
from collections import defaultdict
from datetime import datetime


def count_files_by_extension(directory):
    """Count files in a directory, grouped by extension."""
    if not os.path.exists(directory):
        return None

    counts = defaultdict(int)
    total = 0

    try:
        for item in Path(directory).iterdir():
            if item.is_file():
                ext = item.suffix.lower() or '(no extension)'
                counts[ext] += 1
                total += 1
    except PermissionError:
        return None

    return {'total': total, 'by_extension': dict(counts)}


def count_subdirectories(directory):
    """Count subdirectories in a directory."""
    if not os.path.exists(directory):
        return None

    try:
        subdirs = [d for d in Path(directory).iterdir() if d.is_dir()]
        return len(subdirs)
    except PermissionError:
        return None


def format_count(count_data):
    """Format count data for display."""
    if count_data is None:
        return "❌ Not found"

    if isinstance(count_data, int):
        return str(count_data)

    total = count_data['total']
    by_ext = count_data['by_extension']

    if not by_ext:
        return "0"

    # Format extensions
    ext_parts = [f"{ext}: {count}" for ext, count in sorted(by_ext.items())]
    return f"{total} ({', '.join(ext_parts)})"


def analyze_processing_log(log_path='processing_log.csv'):
    """
    Analyze processing log to extract timing and page count information.

    Returns dict with:
        - total_pages: Total pages processed
        - total_documents: Number of unique documents processed
        - start_time: First timestamp in log
        - end_time: Last timestamp in log
        - elapsed_seconds: Real time elapsed
        - successful_conversions: Number of successful conversions
        - failed_conversions: Number of failed conversions
    """
    if not os.path.exists(log_path):
        return None

    try:
        timestamps = []
        documents = set()
        total_pages = 0
        successful_conversions = 0
        failed_conversions = 0

        with open(log_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Parse timestamp
                try:
                    ts = datetime.fromisoformat(row['timestamp'])
                    timestamps.append(ts)
                except (ValueError, KeyError):
                    continue

                # Track unique documents
                doc_id = row.get('doc_id', '')
                if doc_id:
                    documents.add(doc_id)

                # Count pages from conversion stage
                stage = row.get('stage', '')
                if stage == 'conversion':
                    try:
                        pages = int(row.get('pages', 0))
                        total_pages += pages

                        # Track success/failure
                        status = row.get('status', '')
                        if status == 'success':
                            successful_conversions += 1
                        elif status == 'failed':
                            failed_conversions += 1
                    except (ValueError, TypeError):
                        pass

        if not timestamps:
            return None

        start_time = min(timestamps)
        end_time = max(timestamps)
        elapsed = (end_time - start_time).total_seconds()

        return {
            'total_pages': total_pages,
            'total_documents': len(documents),
            'successful_conversions': successful_conversions,
            'failed_conversions': failed_conversions,
            'start_time': start_time,
            'end_time': end_time,
            'elapsed_seconds': elapsed,
            'elapsed_minutes': elapsed / 60,
            'elapsed_hours': elapsed / 3600,
        }

    except Exception as e:
        print(f"Error analyzing log: {e}")
        return None


def main():
    """Print file counts for all important directories."""

    # Define directories to check
    directories = {
        'PDFs to Process': 'data/to_process',
        'Processed Markdown': 'data/processed',
        'Raw Markdown': 'data/processed_raw',
        'Processed PDFs (archive)': 'data/pdfs_processed',
        'Images': 'data/images',
        'Generated Files': '.generated',
    }

    print("=" * 70)
    print("PDF PIPELINE - FILE COUNTS")
    print("=" * 70)
    print()

    # Print file counts for main directories
    for label, directory in directories.items():
        if directory == 'images':
            # Special handling for images directory
            subdir_count = count_subdirectories(directory)
            if subdir_count is not None:
                print(f"{label:30} {subdir_count} document folders")

                # Count total images across all subdirectories
                total_images = 0
                if os.path.exists(directory):
                    for doc_dir in Path(directory).iterdir():
                        if doc_dir.is_dir():
                            img_count = count_files_by_extension(doc_dir)
                            if img_count:
                                total_images += img_count['total']
                print(f"{'  └─ Total images':30} {total_images}")
            else:
                print(f"{label:30} ❌ Not found")
        elif directory == '.generated':
            # Special handling for .generated directory
            if os.path.exists(directory):
                print(f"{label:30}")
                subdirs = [
                    'image_description_batches',
                    'filter_batches',
                    'upload_batches',
                ]
                for subdir in subdirs:
                    subdir_path = os.path.join(directory, subdir)
                    count = count_files_by_extension(subdir_path)
                    print(f"  └─ {subdir:26} {format_count(count)}")
            else:
                print(f"{label:30} ❌ Not found")
        else:
            count = count_files_by_extension(directory)
            print(f"{label:30} {format_count(count)}")

    print()

    # Check for metadata database
    db_path = 'data/to_process/metadata.db'
    if os.path.exists(db_path):
        db_size = os.path.getsize(db_path) / (1024 * 1024)  # MB
        print(f"{'Metadata Database':30} ✓ ({db_size:.2f} MB)")
    else:
        print(f"{'Metadata Database':30} ❌ Not found")

    # Check for processing log
    log_path = 'processing_log.csv'
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            line_count = sum(1 for _ in f) - 1  # Subtract header
        print(f"{'Processing Log Entries':30} {line_count}")
    else:
        print(f"{'Processing Log':30} ❌ Not found")

    print()
    print("=" * 70)

    # Analyze processing log for detailed statistics
    log_stats = analyze_processing_log(log_path)
    if log_stats:
        print("PROCESSING STATISTICS (from log)")
        print("=" * 70)
        print()

        # Documents processed
        print(f"{'Documents Processed':30} {log_stats['total_documents']}")
        print(f"  └─ Successful conversions:    {log_stats['successful_conversions']}")
        if log_stats['failed_conversions'] > 0:
            print(f"  └─ Failed conversions:        {log_stats['failed_conversions']}")

        # Pages processed
        print(f"\n{'Total Pages Processed':30} {log_stats['total_pages']:,}")
        if log_stats['successful_conversions'] > 0:
            avg_pages = log_stats['total_pages'] / log_stats['successful_conversions']
            print(f"  └─ Average pages/document:    {avg_pages:.1f}")

        # Time elapsed
        print(f"\n{'Processing Period':30}")
        print(f"  └─ Start:                     {log_stats['start_time'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  └─ End:                       {log_stats['end_time'].strftime('%Y-%m-%d %H:%M:%S')}")

        # Format elapsed time
        elapsed_hours = int(log_stats['elapsed_hours'])
        elapsed_minutes = int(log_stats['elapsed_minutes'] % 60)
        elapsed_seconds = int(log_stats['elapsed_seconds'] % 60)

        print(f"\n{'Total Time Elapsed':30} ", end='')
        if elapsed_hours > 0:
            print(f"{elapsed_hours}h {elapsed_minutes}m {elapsed_seconds}s")
        elif elapsed_minutes > 0:
            print(f"{elapsed_minutes}m {elapsed_seconds}s")
        else:
            print(f"{elapsed_seconds}s")

        # Processing rate
        if log_stats['elapsed_minutes'] > 0:
            pages_per_minute = log_stats['total_pages'] / log_stats['elapsed_minutes']
            docs_per_minute = log_stats['successful_conversions'] / log_stats['elapsed_minutes']
            print(f"  └─ Pages per minute:          {pages_per_minute:.1f}")
            print(f"  └─ Documents per minute:      {docs_per_minute:.2f}")

        print()
        print("=" * 70)


if __name__ == '__main__':
    main()
