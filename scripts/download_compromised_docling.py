#!/usr/bin/env python3
"""
Download compromised docling files from S3 bucket.

This script reads document IDs from a CSV file and downloads the corresponding
docling files from the AWS S3 bucket to the local data/processed directory.

Usage:
    python3 scripts/download_compromised_docling.py document_locations_v2_rows.csv
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import List, Dict
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from tqdm import tqdm

def load_document_ids(csv_file: Path) -> List[Dict]:
    """Load document IDs and S3 paths from CSV file."""
    documents = []
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('representation_type') == 'DOCLING':
                documents.append({
                    'document_id': row['kdocument_id'],
                    's3_path': row['s3_key'],
                    'file_size': int(row.get('content_length', 0)),
                    'checksum': row.get('checksum', '')
                })
    
    return documents

def download_docling_file(s3_client, bucket: str, s3_path: str, local_path: Path) -> bool:
    """Download a single docling file from S3."""
    try:
        # Ensure local directory exists
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Download the file
        s3_client.download_file(bucket, s3_path, str(local_path))
        return True
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == '404':
            print(f"⚠️  File not found: s3://{bucket}/{s3_path}")
        else:
            print(f"❌ S3 error downloading {s3_path}: {e}")
        return False
        
    except Exception as e:
        print(f"❌ Error downloading {s3_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Download compromised docling files from S3')
    parser.add_argument('csv_file', type=Path, help='CSV file with document locations')
    parser.add_argument('--bucket', default='primer-production-librarian-documents', 
                       help='S3 bucket name (default: primer-production-librarian-documents)')
    parser.add_argument('--output-dir', type=Path, default=Path('data/processed'),
                       help='Output directory (default: data/processed)')
    parser.add_argument('--profile', default=None,
                       help='AWS profile to use (default: default)')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Show what would be downloaded without actually downloading')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.csv_file.exists():
        print(f"❌ CSV file not found: {args.csv_file}")
        return 1
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🔍 Loading document IDs from CSV...")
    documents = load_document_ids(args.csv_file)
    
    if not documents:
        print("❌ No DOCLING documents found in CSV")
        return 1
    
    print(f"📊 Found {len(documents)} DOCLING documents to download")
    
    if args.dry_run:
        print("\n🔍 DRY RUN - Files that would be downloaded:")
        total_size = 0
        for doc in documents:
            local_path = args.output_dir / f"doc_{doc['document_id']}.txt"
            print(f"  {doc['s3_path']} -> {local_path}")
            total_size += doc['file_size']
        print(f"\n📊 Total files: {len(documents)}")
        print(f"📊 Total size: {total_size:,} bytes ({total_size / 1024 / 1024:.1f} MB)")
        return 0
    
    # Initialize S3 client
    try:
        if args.profile:
            session = boto3.Session(profile_name=args.profile)
            s3_client = session.client('s3')
            print(f"✅ S3 client initialized with profile: {args.profile}")
        else:
            s3_client = boto3.client('s3')
            print("✅ S3 client initialized with default credentials")
    except NoCredentialsError:
        print("❌ AWS credentials not found. Please configure AWS credentials.")
        return 1
    except Exception as e:
        print(f"❌ Error initializing S3 client: {e}")
        return 1
    
    # Download files
    print(f"\n📥 Downloading {len(documents)} files to {args.output_dir}...")
    
    successful_downloads = 0
    failed_downloads = 0
    total_size = 0
    
    with tqdm(total=len(documents), desc="Downloading") as pbar:
        for doc in documents:
            # Create local file path
            local_path = args.output_dir / f"doc_{doc['document_id']}.txt"
            
            # Skip if file already exists
            if local_path.exists():
                pbar.set_description(f"Skipping existing: doc_{doc['document_id']}")
                successful_downloads += 1
                total_size += local_path.stat().st_size
            else:
                pbar.set_description(f"Downloading: doc_{doc['document_id']}")
                
                if download_docling_file(s3_client, args.bucket, doc['s3_path'], local_path):
                    successful_downloads += 1
                    total_size += local_path.stat().st_size
                else:
                    failed_downloads += 1
            
            pbar.update(1)
    
    # Summary
    print(f"\n✅ Download complete!")
    print(f"   Successful: {successful_downloads}")
    print(f"   Failed: {failed_downloads}")
    print(f"   Total size: {total_size:,} bytes ({total_size / 1024 / 1024:.1f} MB)")
    print(f"   Files saved to: {args.output_dir}")
    
    if failed_downloads > 0:
        print(f"\n⚠️  {failed_downloads} files failed to download. Check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
