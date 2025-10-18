import asyncio
import aioboto3
from pathlib import Path

async def download_test():
    # Initialize S3 client
    session = aioboto3.Session(profile_name="production", region_name="eu-west-2")
    async with session.client("s3") as s3_client:
        bucket = "primer-production-librarian-documents"
        
        # Download both files for doc_27355
        files_to_download = [
            ("documents/27355/representations/docling/doc_27355.txt", "test_download_docling.txt"),
            ("documents/27355/representations/docling_img/doc_27355.txt", "test_download_docling_img.txt"),
        ]
        
        for s3_key, local_file in files_to_download:
            print(f"📥 Downloading: {s3_key}")
            try:
                response = await s3_client.get_object(Bucket=bucket, Key=s3_key)
                content = await response['Body'].read()
                
                # Save to local file
                Path(local_file).write_bytes(content)
                
                # Show first 500 characters
                text_preview = content.decode('utf-8')[:500]
                print(f"✅ Downloaded {len(content)} bytes")
                print(f"📄 Preview:\n{text_preview}...\n")
                
            except Exception as e:
                print(f"❌ Error: {e}\n")

asyncio.run(download_test())
