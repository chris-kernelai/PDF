#!/usr/bin/env python3
"""
Batch upload script for document representations from data/processed and data/processed_images.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
from standalone_upload_representations import DocumentRepresentationUploader


async def batch_upload_documents(
    processed_dir: str = "data/processed",
    processed_images_dir: str = "data/processed_images",
    limit: int = None,
    dry_run: bool = False,
):
    """
    Batch upload all documents from processed and processed_images directories.

    Args:
        processed_dir: Path to directory containing docling files
        processed_images_dir: Path to directory containing docling_img files
        limit: Optional limit on number of documents to process
        dry_run: If True, only print what would be uploaded without actually uploading
    """

    processed_path = Path(processed_dir)
    processed_images_path = Path(processed_images_dir)

    if not processed_path.exists():
        print(f"❌ Directory not found: {processed_dir}")
        return

    if not processed_images_path.exists():
        print(f"⚠️  Directory not found: {processed_images_dir}")
        print(f"   Will upload only raw markdown files (no image processing)")
        processed_images_files = {}  # Empty dict for no image files
    else:
        processed_images_files = {}

    # Get all markdown files and extract document IDs
    processed_files = {}
    for file in processed_path.glob("doc_*.md"):
        # Extract document ID from filename (e.g., doc_27290.md -> 27290)
        doc_id_str = file.stem.replace("doc_", "")
        try:
            doc_id = int(doc_id_str)
            processed_files[doc_id] = file
        except ValueError:
            print(f"⚠️  Skipping invalid filename: {file.name}")
            continue

    # Get all processed_images files (if directory exists)
    if processed_images_path.exists():
        for file in processed_images_path.glob("doc_*.md"):
            doc_id_str = file.stem.replace("doc_", "")
            try:
                doc_id = int(doc_id_str)
                processed_images_files[doc_id] = file
            except ValueError:
                print(f"⚠️  Skipping invalid filename: {file.name}")
                continue

    # Find documents to upload
    if processed_images_files:
        # Both types of files exist - find matching pairs
        matching_doc_ids = set(processed_files.keys()) & set(processed_images_files.keys())
    else:
        # Only raw markdown files exist - upload all of them
        matching_doc_ids = set(processed_files.keys())

    print(f"\n📊 File Summary:")
    print(f"   Total docling files: {len(processed_files)}")
    if processed_images_files:
        print(f"   Total docling_img files: {len(processed_images_files)}")
        print(f"   Matching pairs: {len(matching_doc_ids)}")
    else:
        print(f"   Total docling_img files: 0 (raw markdown only)")
        print(f"   Documents to upload: {len(matching_doc_ids)}")

    # Initialize uploader early to check existing uploads
    uploader = DocumentRepresentationUploader()

    try:
        await uploader.initialize()

        # Check which documents already have representations uploaded
        print(f"\n🔍 Checking for existing uploads...")
        existing_reps = await uploader.get_existing_representations(
            document_ids=list(matching_doc_ids)
        )

        # Filter out documents that already have BOTH representations
        docs_to_skip = set()
        docs_partial = set()

        for doc_id in matching_doc_ids:
            if doc_id in existing_reps:
                reps = existing_reps[doc_id]
                if "DOCLING" in reps and "DOCLING_IMG" in reps:
                    # Both representations exist, skip entirely
                    docs_to_skip.add(doc_id)
                else:
                    # Only partial upload exists
                    docs_partial.add(doc_id)

        # Documents to upload: those without both representations
        docs_to_upload = matching_doc_ids - docs_to_skip

        print(f"\n📊 Upload Plan:")
        print(f"   ✅ Already uploaded (skipping): {len(docs_to_skip)}")
        print(f"   ⚠️  Partial uploads (will complete): {len(docs_partial)}")
        print(f"   📤 New uploads needed: {len(docs_to_upload) - len(docs_partial)}")
        print(f"   📋 Total to process: {len(docs_to_upload)}")

        if limit:
            docs_to_upload = set(sorted(docs_to_upload)[:limit])
            print(f"   🔢 Processing (limited): {len(docs_to_upload)}")

        if dry_run:
            print(f"\n🔍 DRY RUN MODE - No uploads will be performed\n")
            for doc_id in sorted(docs_to_upload)[:10]:  # Show first 10
                status = ""
                if doc_id in docs_partial:
                    status = " (partial - will complete)"
                print(f"   Would upload doc_{doc_id}{status}:")
                print(f"      docling: {processed_files[doc_id]}")
                if processed_images_files.get(doc_id):
                    print(f"      docling_img: {processed_images_files[doc_id]}")
                else:
                    print(f"      docling_img: (raw markdown only)")
            if len(docs_to_upload) > 10:
                print(f"   ... and {len(docs_to_upload) - 10} more")
            await uploader.close()
            return

        if len(docs_to_upload) == 0:
            print(f"\n✅ All documents already uploaded! Nothing to do.")
            await uploader.close()
            return

        success_count = 0
        error_count = 0
        skipped_count = 0
        errors_log = []

        print(f"\n🚀 Starting upload of {len(docs_to_upload)} documents...\n")

        for i, doc_id in enumerate(sorted(docs_to_upload), 1):
            docling_file = processed_files[doc_id]
            docling_img_file = processed_images_files.get(doc_id)  # May be None

            status_str = ""
            if doc_id in docs_partial:
                status_str = " (completing partial)"

            print(f"[{i}/{len(docs_to_upload)}] Processing doc_{doc_id}{status_str}...")

            try:
                # Check if individual representations already exist
                existing = existing_reps.get(doc_id, set())

                # Skip individual files if they already exist
                upload_docling = "DOCLING" not in existing
                upload_docling_img = docling_img_file and "DOCLING_IMG" not in existing

                if not upload_docling and not upload_docling_img:
                    # Shouldn't happen due to filtering, but just in case
                    skipped_count += 1
                    print(f"   ⏭️  Already exists, skipping")
                    continue

                # Upload the representations (with individual checks)
                results = await uploader.upload_representations(
                    document_id=doc_id,
                    docling_file=str(docling_file) if upload_docling else None,
                    docling_img_file=str(docling_img_file) if upload_docling_img else None,
                    docling_filename=f"doc_{doc_id}.txt",
                    docling_img_filename=f"doc_{doc_id}.txt" if docling_img_file else None,
                )

                if results["errors"]:
                    error_count += 1
                    error_msg = f"doc_{doc_id}: {', '.join(results['errors'])}"
                    errors_log.append(error_msg)
                    print(f"   ❌ Partial failure: {error_msg}")
                else:
                    success_count += 1
                    print(f"   ✅ Success")

            except Exception as e:
                error_count += 1
                error_msg = f"doc_{doc_id}: {str(e)}"
                errors_log.append(error_msg)
                print(f"   ❌ Error: {e}")

        # Print final summary
        print(f"\n" + "="*60)
        print(f"📊 UPLOAD COMPLETE")
        print(f"="*60)
        print(f"   ✅ Successful: {success_count}")
        print(f"   ❌ Failed: {error_count}")
        print(f"   ⏭️  Already uploaded (skipped): {len(docs_to_skip)}")
        if len(docs_to_upload) > 0:
            print(f"   📈 Success rate: {success_count}/{len(docs_to_upload)} ({100*success_count/len(docs_to_upload):.1f}%)")

        if errors_log:
            print(f"\n❌ Errors:")
            for error in errors_log[:10]:  # Show first 10 errors
                print(f"   - {error}")
            if len(errors_log) > 10:
                print(f"   ... and {len(errors_log) - 10} more errors")

    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        await uploader.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch upload document representations to S3")
    parser.add_argument(
        "--processed-dir",
        default="data/processed",
        help="Directory containing docling files (default: data/processed)"
    )
    parser.add_argument(
        "--processed-images-dir",
        default="data/processed_images",
        help="Directory containing docling_img files (default: data/processed_images)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of documents to process (useful for testing)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be uploaded without actually uploading"
    )

    args = parser.parse_args()

    asyncio.run(batch_upload_documents(
        processed_dir=args.processed_dir,
        processed_images_dir=args.processed_images_dir,
        limit=args.limit,
        dry_run=args.dry_run,
    ))
