#!/usr/bin/env python3
"""
Count files in important pipeline directories.
"""
import os
from pathlib import Path
from collections import defaultdict


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


if __name__ == '__main__':
    main()
