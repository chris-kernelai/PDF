#!/usr/bin/env python3
"""
Upload only docling_img files to replace compromised ones.

This script uploads the newly generated docling_img files to S3,
replacing only the docling_img representations while leaving
the original docling files unchanged.

Usage:
    python3 scripts/upload_docling_img_only.py --session-id abc123
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from tqdm import tqdm

def load_upload_config() -> Dict:
    """Load upload configuration from config.yaml"""
    try:
        import yaml
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        return config.get('upload', {})
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return {}

def verify_files_exist_in_s3(s3_client: object, bucket_name: str, base_path: str, doc_ids: List[str], sample_size: int = 100) -> bool:
    """Verify that files exist in S3 before attempting to replace them"""
    import random
    
    # Sample random document IDs
    sample_doc_ids = random.sample(doc_ids, min(sample_size, len(doc_ids)))
    
    print(f"🔍 Verifying {len(sample_doc_ids)} random files exist in S3...")
    
    existing_count = 0
    missing_count = 0
    
    for doc_id in tqdm(sample_doc_ids, desc="Verifying"):
        s3_key = f"{base_path}/{doc_id}/representations/docling_img/doc_{doc_id}.txt"
        
        try:
            s3_client.head_object(Bucket=bucket_name, Key=s3_key)
            existing_count += 1
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                missing_count += 1
                print(f"⚠️  File not found: {s3_key}")
            else:
                print(f"❌ Error checking {s3_key}: {e}")
                missing_count += 1
    
    print(f"\n📊 Verification Results:")
    print(f"   ✅ Found: {existing_count}")
    print(f"   ❌ Missing: {missing_count}")
    print(f"   📊 Total checked: {len(sample_doc_ids)}")
    
    if missing_count > 0:
        print(f"\n⚠️  {missing_count} files are missing from S3. This might be expected if they were never uploaded.")
        response = input("Do you want to continue with the upload? (y/N): ")
        if response.lower() != 'y':
            return False
    
    return True

def upload_docling_img_files(session_id: str, profile: str = None) -> bool:
    """Upload docling_img files to S3"""
    
    # Initialize S3 client
    try:
        if profile:
            session = boto3.Session(profile_name=profile)
            s3_client = session.client('s3')
            print(f"✅ S3 client initialized with profile: {profile}")
        else:
            s3_client = boto3.client('s3')
            print("✅ S3 client initialized with default credentials")
    except NoCredentialsError:
        print("❌ AWS credentials not found. Please configure AWS credentials.")
        return False
    except Exception as e:
        print(f"❌ Error initializing S3 client: {e}")
        return False
    
    # Load configuration
    config = load_upload_config()
    bucket_name = config.get('bucket_name', 'primer-production-librarian-documents')
    base_path = config.get('base_path', 'documents')
    
    print(f"📦 Bucket: {bucket_name}")
    print(f"📁 Base path: {base_path}")
    
    # Find all docling_img files
    processed_dir = Path('data/processed')
    docling_img_files = list(processed_dir.glob('doc_*.txt'))
    
    if not docling_img_files:
        print("❌ No docling_img files found in data/processed/")
        return False
    
    print(f"📄 Found {len(docling_img_files)} docling_img files to upload")
    
    # Extract document IDs
    doc_ids = [f.stem.replace('doc_', '') for f in docling_img_files]
    
    # Verify files exist in S3 before proceeding
    if not verify_files_exist_in_s3(s3_client, bucket_name, base_path, doc_ids):
        print("❌ Verification failed or cancelled by user")
        return False
    
    # Confirm before proceeding with replacement
    print(f"\n⚠️  WARNING: This will REPLACE {len(docling_img_files)} docling_img files in S3!")
    print(f"   Bucket: {bucket_name}")
    print(f"   Path pattern: {base_path}/{{doc_id}}/representations/docling_img/doc_{{doc_id}}.txt")
    print(f"   Session ID: {session_id}")
    
    response = input("\nAre you sure you want to proceed with the replacement? (yes/NO): ")
    if response.lower() != 'yes':
        print("❌ Upload cancelled by user")
        return False
    
    # Upload files
    successful_uploads = 0
    failed_uploads = 0
    
    print(f"\n📤 Uploading docling_img files...")
    
    for file_path in tqdm(docling_img_files, desc="Uploading"):
        try:
            # Extract document ID from filename (doc_12345.txt -> 12345)
            doc_id = file_path.stem.replace('doc_', '')
            
            # Construct S3 key for docling_img
            s3_key = f"{base_path}/{doc_id}/representations/docling_img/doc_{doc_id}.txt"
            
            # Check if file exists before replacing
            try:
                s3_client.head_object(Bucket=bucket_name, Key=s3_key)
                print(f"🔄 Replacing existing file: {s3_key}")
            except ClientError as e:
                if e.response['Error']['Code'] == '404':
                    print(f"➕ Creating new file: {s3_key}")
                else:
                    print(f"⚠️  Could not check existing file {s3_key}: {e}")
            
            # Upload file
            s3_client.upload_file(
                str(file_path),
                bucket_name,
                s3_key,
                ExtraArgs={
                    'ContentType': 'text/plain',
                    'Metadata': {
                        'session-id': session_id,
                        'upload-type': 'docling_img_replacement',
                        'original-filename': file_path.name
                    }
                }
            )
            
            successful_uploads += 1
            
        except ClientError as e:
            print(f"❌ Error uploading {file_path.name}: {e}")
            failed_uploads += 1
        except Exception as e:
            print(f"❌ Unexpected error uploading {file_path.name}: {e}")
            failed_uploads += 1
    
    # Summary
    print(f"\n📊 Upload Summary:")
    print(f"   ✅ Successful: {successful_uploads}")
    print(f"   ❌ Failed: {failed_uploads}")
    print(f"   📦 Total: {len(docling_img_files)}")
    
    if failed_uploads > 0:
        print(f"\n⚠️  {failed_uploads} files failed to upload")
        return False
    
    print(f"\n🎉 All docling_img files uploaded successfully!")
    print(f"🔑 Session ID: {session_id}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Upload only docling_img files to replace compromised ones')
    parser.add_argument('--session-id', required=True, help='Session ID for this upload')
    parser.add_argument('--profile', help='AWS profile to use (default: default)')
    
    args = parser.parse_args()
    
    print(f"🚀 Starting docling_img upload for session: {args.session_id}")
    
    success = upload_docling_img_files(args.session_id, args.profile)
    
    if success:
        print("✅ Upload completed successfully")
        sys.exit(0)
    else:
        print("❌ Upload failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
