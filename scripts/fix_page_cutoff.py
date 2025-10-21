#!/usr/bin/env python3
"""
Fix page cutoff issues for documents processed with 30-page chunking.

This script:
1. Downloads DOCLING representations from AWS using Supabase table
2. Identifies and fixes missing pages that are multiples of 30 (30, 60, 90, etc.) plus the last page
3. Reuploads the fixed files

The issue: When chunking at 30 pages, the last page of each chunk was being cut off.
This script fixes those specific pages by re-extracting them from the original PDFs.
"""

import argparse
import asyncio
import csv
import os
import re
import sys
from pathlib import Path
from typing import Dict, Optional, Set

import aiohttp
import aiofiles
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from dotenv import load_dotenv
from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from docling_converter import DoclingConverter
from standalone_upload_representations import DocumentRepresentationUploader

# Load environment variables
load_dotenv()

# Constants
S3_BUCKET_DEFAULT = "primer-production-librarian-documents"
DEFAULT_CSV_PATH = Path("document_locations_v2_rows.csv")
DOWNLOAD_DIR = Path("scripts/data/fix_download")
UPLOAD_DIR = Path("scripts/data/fix_upload")


class PageCutoffFixer:
    """Fixes page cutoff issues in DOCLING representations."""
    
    def __init__(self, aws_profile: Optional[str] = None, dry_run: bool = False, verbose: bool = False):
        # Use IAM role if no explicit credentials, otherwise use profile
        if os.getenv("AWS_ACCESS_KEY_ID"):
            self.aws_profile = None  # Use IAM role or explicit credentials
        elif aws_profile:
            self.aws_profile = aws_profile
        elif os.getenv("AWS_PROFILE"):
            self.aws_profile = os.getenv("AWS_PROFILE")
        else:
            self.aws_profile = None  # Use IAM role by default
        self.dry_run = dry_run
        self.verbose = verbose
        self.s3_client = None
        self.uploader = None
        self.api_key = os.getenv("API_KEY")
        self.base_url = os.getenv("API_BASE_URL", "https://librarian.production.primerapp.com/api/v1")
        self.aws_region = os.getenv("AWS_REGION", "eu-west-2")
        
    async def initialize(self):
        """Initialize AWS and database connections."""
        try:
            # Initialize S3 client
            if self.aws_profile:
                session = boto3.Session(profile_name=self.aws_profile)
                self.s3_client = session.client("s3", region_name=self.aws_region)
            else:
                # Use IAM role or default credential chain
                self.s3_client = boto3.client("s3", region_name=self.aws_region)
            
            # Only initialize uploader if not in dry run mode
            if not self.dry_run:
                self.uploader = DocumentRepresentationUploader(aws_profile=self.aws_profile)
                await self.uploader.initialize()
            
        except NoCredentialsError:
            if self.aws_profile:
                print(f"❌ AWS credentials not found for profile '{self.aws_profile}'. If using SSO, run: aws sso login --profile {self.aws_profile}")
            else:
                print("❌ AWS credentials not found. Check IAM role or run: aws configure")
            raise
        except Exception as e:
            print(f"❌ Failed to initialize: {e}")
            raise
    
    def load_documents_from_csv(self, csv_path: Path) -> Dict[int, Dict[str, Dict[str, str]]]:
        """Load document information from CSV file."""
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        
        documents: Dict[int, Dict[str, Dict[str, str]]] = {}
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                doc_id = int(row["kdocument_id"])
                rep_type = row["representation_type"]
                
                if rep_type == "DOCLING":
                    if doc_id not in documents:
                        documents[doc_id] = {}
                    documents[doc_id]["docling"] = {
                        "s3_key": row["s3_key"],
                        "bucket": row["s3_bucket"] or S3_BUCKET_DEFAULT,
                        "page_count": str(int(row["page_count"])) if row["page_count"] else "0"
                    }
                elif rep_type == "DOCLING_IMG":
                    if doc_id not in documents:
                        documents[doc_id] = {}
                    documents[doc_id]["docling_img"] = {
                        "s3_key": row["s3_key"],
                        "bucket": row["s3_bucket"] or S3_BUCKET_DEFAULT,
                        "page_count": str(int(row["page_count"])) if row["page_count"] else "0"
                    }
        
        return documents
    
    async def download_document(self, doc_id: int, doc_info: Dict[str, Dict[str, str]]) -> Optional[Path]:
        """Download a DOCLING document from S3."""
        if "docling" not in doc_info:
            return None
        
        # Create download path
        download_path = DOWNLOAD_DIR / f"doc_{doc_id}.md"
        download_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if file already exists
        if download_path.exists():
            if self.verbose:
                print(f"📂 Using cached markdown for doc {doc_id}")
            return download_path
        
        s3_key = doc_info["docling"]["s3_key"]
        bucket = doc_info["docling"]["bucket"]
        
        try:
            if self.s3_client:
                self.s3_client.download_file(bucket, s3_key, str(download_path))
                return download_path
            else:
                return None
        except ClientError as e:
            if self.verbose:
                print(f"❌ Failed to download doc {doc_id}: {e}")
            return None
    
    async def _get_download_urls_batch(self, session: aiohttp.ClientSession, document_ids: list[int]) -> Dict[int, Optional[str]]:
        """Get presigned download URLs for a batch of document IDs."""
        if not self.api_key:
            if self.verbose:
                print("❌ API_KEY not set, cannot download PDFs")
            return {}
        
        payload = {
            "documents": [
                {
                    "document_id": doc_id,
                    "representation_type": "raw",  # Use raw to get original PDF
                    "expires_in": 3600,  # URL valid for 1 hour
                }
                for doc_id in document_ids
            ]
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            async with session.post(
                f"{self.base_url}/kdocuments/batch/download",
                json=payload,
                headers=headers
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    results = data.get("data", {}).get("results", [])
                    
                    url_map = {}
                    for result in results:
                        doc_id = result.get("document_id")
                        download_url = result.get("download_url")
                        error = result.get("error")
                        
                        if error:
                            if self.verbose:
                                print(f"⚠️  Document {doc_id} download failed: {error}")
                            url_map[doc_id] = None
                        else:
                            url_map[doc_id] = download_url
                    
                    return url_map
                else:
                    if self.verbose:
                        print(f"❌ Failed to get download URLs: HTTP {response.status}")
                    return {}
        except Exception as e:
            if self.verbose:
                print(f"❌ Error getting download URLs: {e}")
            return {}
    
    async def _download_pdf(self, session: aiohttp.ClientSession, document_id: int, pdf_url: str) -> Optional[Path]:
        """Download a single PDF file from a presigned URL."""
        if not pdf_url:
            return None
        
        pdf_filename = f"doc_{document_id}.pdf"
        pdf_path = DOWNLOAD_DIR / pdf_filename
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if file already exists
        if pdf_path.exists():
            if self.verbose:
                print(f"📂 Using cached PDF for doc {document_id}")
            return pdf_path
        
        try:
            async with session.get(pdf_url) as response:
                if response.status == 200:
                    async with aiofiles.open(pdf_path, "wb") as f:
                        await f.write(await response.read())
                    
                    return pdf_path
                else:
                    if self.verbose:
                        print(f"❌ Failed to download PDF for doc {document_id}: HTTP {response.status}")
                    return None
        except Exception as e:
            if self.verbose:
                print(f"❌ Error downloading PDF for doc {document_id}: {e}")
            return None
    
    def find_missing_pages(self, markdown_content: str, total_pages: int) -> Set[int]:
        """Find ALL missing page numbers by comparing against expected pages 1 to total_pages."""
        # Find all page markers in the content
        page_markers = re.findall(r'<!-- PAGE (\d+) -->', markdown_content)
        existing_pages = {int(page) for page in page_markers}
        
        if not existing_pages:
            return set()
        
        # Find ALL missing pages by comparing against expected range
        expected_pages = set(range(1, total_pages + 1))
        missing_pages = expected_pages - existing_pages
        
        return missing_pages
    
    def extract_page_content_from_pdf(self, pdf_path: Path, page_num: int, converter: DoclingConverter) -> str:
        """Extract content for a specific page from PDF using DoclingConverter."""
        try:
            # Create a temporary PDF with just the target page
            from pypdf import PdfReader, PdfWriter
            
            reader = PdfReader(str(pdf_path))
            writer = PdfWriter()
            
            # Add the specific page (page_num is 1-based, but PDFReader uses 0-based)
            writer.add_page(reader.pages[page_num - 1])
            
            # Save the single-page PDF for debugging (optional)
            # single_page_pdf = DOWNLOAD_DIR / f"page_{page_num}_extracted.pdf"
            # single_page_pdf.parent.mkdir(parents=True, exist_ok=True)
            # 
            # with open(single_page_pdf, "wb") as f:
            #     writer.write(f)
            # 
            # print(f"    💾 Saved single-page PDF: {single_page_pdf}")
            
            # Check what we actually extracted (using temporary file)
            import tempfile as tmp
            import os as os_module
            with tmp.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                writer.write(temp_file)
                temp_file.flush()
                
                # Convert the single-page PDF using the direct export method
                markdown, _, _ = converter.convert_pdf(temp_file.name, page_offset=0)
                
                # Clean up temp file
                os_module.unlink(temp_file.name)
            
            # Extract just the page content (remove page markers and "Page X" headers)
            # The markdown from convert_pdf includes page markers which we don't want
            import re as re_module
            # Remove "Page X" headers
            markdown = re_module.sub(r'^Page \d+\s*\n', '', markdown, flags=re_module.MULTILINE)
            # Remove page markers
            markdown = re_module.sub(r'<!-- PAGE \d+ -->', '', markdown)
            # Clean up extra whitespace
            markdown = markdown.strip()
            
            # print(f"    🔍 Direct export output: {len(markdown)} chars")
            # print(f"    🔍 Direct export preview: {repr(markdown[:200])}")
            
            # Save the extracted content to a separate file for debugging (optional)
            # output_file = DOWNLOAD_DIR / f"page_{page_num}_extracted.md"
            # with open(output_file, 'w', encoding='utf-8') as f:
            #     f.write(markdown)
            # print(f"    💾 Saved extracted content to: {output_file}")
            
            return markdown
                
        except (FileNotFoundError, ValueError, RuntimeError) as e:
            print(f"❌ Failed to extract page {page_num} from PDF: {e}")
            return ""
    
    def insert_missing_pages(self, markdown_content: str, missing_pages: Set[int], pdf_path: Path, total_pages: int) -> str:
        """Insert missing pages into the markdown content."""
        if not missing_pages:
            return markdown_content
        
        # Create a single converter instance to reuse
        converter = DoclingConverter()
        
        try:
            # Extract all missing pages first
            extracted_pages = {}
            for page_num in sorted(missing_pages):
                page_content = self.extract_page_content_from_pdf(pdf_path, page_num, converter)
                if page_content:
                    extracted_pages[page_num] = page_content
            
            if not extracted_pages:
                return markdown_content
            
            # Now insert the extracted pages into the content
            # Find all page markers and their positions
            page_markers = list(re.finditer(r'<!-- PAGE (\d+) -->', markdown_content))
            
            if not page_markers:
                return markdown_content
            
            # Build the fixed content by inserting missing pages
            # Create a list of all pages (existing + missing) in order
            all_pages = []
            
            # Get all existing page numbers
            existing_pages = [int(match.group(1)) for match in page_markers]
            all_pages.extend(existing_pages)
            
            # Add missing pages
            all_pages.extend(sorted(missing_pages))
            
            # Sort all pages
            all_pages.sort()
            
            # Simple approach: insert missing pages at the right positions
            result_content = markdown_content
            
            # Insert missing pages in order
            for page_num in sorted(missing_pages):
                if page_num in extracted_pages:
                    # Look for the next page marker (current page + 1)
                    next_page = page_num + 1
                    next_page_pattern = f'<!-- PAGE {next_page} -->'
                    next_page_pos = result_content.find(next_page_pattern)
                    
                    if next_page_pos != -1:
                        # Found the next page marker - insert BEFORE it
                        result_content = (result_content[:next_page_pos] + 
                                       f"\n\n<!-- PAGE {page_num} -->\n\n{extracted_pages[page_num]}\n\n" +
                                       result_content[next_page_pos:])
                    else:
                        # No next page marker found - this is the last page, put it at the end
                        result_content += f"\n\n<!-- PAGE {page_num} -->\n\n{extracted_pages[page_num]}"
            
            return result_content
        
        except Exception as e:
            if self.verbose:
                print(f"❌ Error inserting missing pages: {e}")
            return markdown_content
        finally:
            # Clean up the converter
            converter.cleanup()
    
    async def fix_document(self, doc_id: int, doc_info: Dict[str, Dict[str, str]], pbar: tqdm) -> dict:
        """Fix a single document by downloading, fixing, and reuploading.
        
        Returns dict with status and details for reporting.
        """
        result = {"doc_id": doc_id, "status": "error", "missing_pages": [], "reprocessed": False}
        
        # Download the DOCLING representation
        markdown_path = await self.download_document(doc_id, doc_info)
        if not markdown_path:
            result["status"] = "skip"
            pbar.update(1)
            return result
        
        # Read the markdown content
        with open(markdown_path, "r", encoding="utf-8") as f:
            markdown_content = f.read()
        
        # Get initial page count from CSV or content analysis
        page_count_str = doc_info["docling"]["page_count"]
        if page_count_str and page_count_str != "0":
            total_pages = int(page_count_str)
        else:
            # Analyze the content to find the highest page number
            page_markers = re.findall(r'<!-- PAGE (\d+) -->', markdown_content)
            if page_markers:
                total_pages = max(int(page) for page in page_markers)
            else:
                result["status"] = "skip"
                pbar.update(1)
                return result
        
        if total_pages <= 0:
            result["status"] = "skip"
            pbar.update(1)
            return result
        
        # Download the original PDF first to get accurate page count
        async with aiohttp.ClientSession() as session:
            url_map = await self._get_download_urls_batch(session, [doc_id])
            pdf_url = url_map.get(doc_id)
            
            if not pdf_url:
                result["status"] = "error"
                pbar.update(1)
                return result
            
            pdf_path = await self._download_pdf(session, doc_id, pdf_url)
            if not pdf_path:
                result["status"] = "error"
                pbar.update(1)
                return result
            
            # Get the actual page count from the PDF
            try:
                from pypdf import PdfReader
                pdf_reader = PdfReader(pdf_path)
                actual_page_count = len(pdf_reader.pages)
                total_pages = actual_page_count
            except Exception as e:
                if self.verbose:
                    print(f"❌ Failed to read PDF page count: {e}")
                # Keep the original total_pages from CSV/content analysis
        
        # Find missing pages with correct total page count
        missing_pages = self.find_missing_pages(markdown_content, total_pages)
        if not missing_pages:
            result["status"] = "ok"
            pbar.update(1)
            return result
        
        result["missing_pages"] = sorted(missing_pages)
        
        # Extract missing pages from PDF and insert them
        fixed_content = self.insert_missing_pages(markdown_content, missing_pages, pdf_path, total_pages)
        
        # Save the fixed content
        fixed_path = UPLOAD_DIR / f"doc_{doc_id}.md"
        fixed_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(fixed_path, "w", encoding="utf-8") as f:
            f.write(fixed_content)
        
        if not self.dry_run:
            # Upload the fixed content back to S3 (replace existing)
            try:
                # Save the fixed content to a temporary file for upload (as .txt)
                temp_file = UPLOAD_DIR / f"temp_doc_{doc_id}.txt"
                with open(temp_file, 'w', encoding='utf-8') as f:
                    f.write(fixed_content)
                
                # Upload using custom replace method
                await self.replace_docling_representation(
                    document_id=doc_id,
                    docling_file=str(temp_file),
                    page_count=total_pages,
                    s3_bucket=doc_info["docling"]["bucket"],
                    s3_key=doc_info["docling"]["s3_key"]
                )
                
                # Clean up temp file
                temp_file.unlink()
                
                result["status"] = "fixed"
                result["reprocessed"] = True
            except Exception as e:
                if self.verbose:
                    print(f"❌ Failed to upload fixed content for doc {doc_id}: {e}")
                result["status"] = "error"
                result["reprocessed"] = False
        else:
            result["status"] = "fixed"
            result["reprocessed"] = True  # Would have been reprocessed
        
        pbar.update(1)
        return result
    
    async def replace_docling_representation(self, document_id: int, docling_file: str, page_count: int, s3_bucket: str, s3_key: str) -> dict:
        """Replace existing DOCLING representation by updating S3 and database."""
        try:
            # Upload file to S3
            s3_metadata = await self.uploader.upload_file_to_s3(docling_file, s3_key)
            
            # Update database record
            async with self.uploader.db_pool.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE librarian.document_locations_v2 
                    SET s3_key = $1, page_count = $2, updated_at = NOW()
                    WHERE kdocument_id = $3 AND representation_type = 'DOCLING'
                    """,
                    s3_key,
                    page_count,
                    document_id
                )
            
            return {
                'document_id': document_id,
                's3_metadata': s3_metadata,
                'status': 'replaced'
            }
            
        except Exception as e:
            if self.verbose:
                print(f"❌ Failed to replace DOCLING representation: {e}")
            raise
    
    async def fix_all_documents(self, csv_path: Path, limit: Optional[int] = None, doc_id: Optional[int] = None):
        """Fix all documents in the CSV file."""
        documents = self.load_documents_from_csv(csv_path)
        
        if doc_id:
            if doc_id in documents:
                documents = {doc_id: documents[doc_id]}
            else:
                print(f"❌ Document {doc_id} not found in CSV")
                return
        elif limit:
            # Process documents in reverse order to get newer ones first
            doc_ids = list(documents.keys())[-limit:]
            documents = {doc_id: documents[doc_id] for doc_id in doc_ids}
        
        # Track results for documents with missing pages
        docs_with_missing_pages = []
        
        # Progress bar
        with tqdm(total=len(documents), desc="Processing documents", unit="doc") as pbar:
            for doc_id, doc_info in documents.items():
                try:
                    result = await self.fix_document(doc_id, doc_info, pbar)
                    
                    # Track documents with missing pages
                    if result["missing_pages"]:
                        docs_with_missing_pages.append(result)
                        
                except (FileNotFoundError, ValueError, RuntimeError, ClientError) as e:
                    if self.verbose:
                        print(f"❌ Error processing doc {doc_id}: {e}")
                    pbar.update(1)
        
        # Report results
        print("\n" + "="*60)
        if docs_with_missing_pages:
            print(f"📋 Documents with missing pages: {len(docs_with_missing_pages)}")
            print("="*60)
            for result in docs_with_missing_pages:
                doc_id = result["doc_id"]
                missing = result["missing_pages"]
                reprocessed = result["reprocessed"]
                status_icon = "✅" if reprocessed else "❌"
                print(f"{status_icon} Doc {doc_id}: Missing pages {missing} - {'Reprocessed' if reprocessed else 'Failed to reprocess'}")
        else:
            print("✅ All documents have complete pages")
        print("="*60)


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Fix page cutoff issues in DOCLING representations")
    parser.add_argument("--csv-file", type=Path, default=DEFAULT_CSV_PATH, help="CSV file path")
    parser.add_argument("--profile", help="AWS profile to use")
    parser.add_argument("--limit", type=int, help="Limit number of documents to process")
    parser.add_argument("--doc-id", type=int, help="Process specific document ID")
    parser.add_argument("--dry-run", action="store_true", help="Don't actually upload, just show what would be done")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Initialize fixer
    fixer = PageCutoffFixer(aws_profile=args.profile, dry_run=args.dry_run, verbose=args.verbose)
    
    try:
        await fixer.initialize()
        await fixer.fix_all_documents(args.csv_file, args.limit, args.doc_id)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
