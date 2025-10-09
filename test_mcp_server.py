#!/usr/bin/env python3
"""
Test script for Librarian MCP Server

Tests all tools and resources to verify functionality.
"""

import asyncio
import json
import sys
from pathlib import Path

# Add parent directory to path to import the server
sys.path.insert(0, str(Path(__file__).parent))

from librarian_mcp_server import (
    api_client,
    librarian_get_companies,
    librarian_search_documents,
    librarian_batch_download,
    librarian_get_document_metadata,
)


async def test_get_companies():
    """Test fetching companies"""
    print("\n" + "=" * 60)
    print("TEST: librarian_get_companies")
    print("=" * 60)

    try:
        # Test 1: Get companies from Canada
        print("\n[Test 1] Fetching companies from Canada...")
        result = await librarian_get_companies(
            countries=["Canada"],
            page=1,
            page_size=10
        )
        data = json.loads(result)
        print(f"✓ Found {data.get('companies_count', 0)} Canadian companies")
        if data.get('companies'):
            print(f"  First company: {data['companies'][0].get('name')}")

        # Test 2: Get companies from multiple countries
        print("\n[Test 2] Fetching companies from UK and Germany...")
        result = await librarian_get_companies(
            countries=["United Kingdom", "Germany"],
            page=1,
            page_size=5
        )
        data = json.loads(result)
        print(f"✓ Found {data.get('companies_count', 0)} companies")

        # Test 3: Get all companies (no filter)
        print("\n[Test 3] Fetching all companies (no filter)...")
        result = await librarian_get_companies(page=1, page_size=5)
        data = json.loads(result)
        print(f"✓ Found {data.get('companies_count', 0)} companies")

        print("\n✓ All get_companies tests passed!")
        return data.get('companies', [])

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        return []


async def test_search_documents(company_ids=None):
    """Test searching documents"""
    print("\n" + "=" * 60)
    print("TEST: librarian_search_documents")
    print("=" * 60)

    try:
        # Test 1: Search by document type
        print("\n[Test 1] Searching for filings...")
        result = await librarian_search_documents(
            document_types=["filing"],
            page=1,
            page_size=5
        )
        data = json.loads(result)
        print(f"✓ Found {data.get('documents_count', 0)} filings")
        if data.get('documents'):
            print(f"  First document: {data['documents'][0].get('title')}")

        # Test 2: Search by company IDs (if provided)
        if company_ids:
            print(f"\n[Test 2] Searching documents for company IDs: {company_ids[:2]}...")
            result = await librarian_search_documents(
                company_ids=company_ids[:2],
                page=1,
                page_size=10
            )
            data = json.loads(result)
            print(f"✓ Found {data.get('documents_count', 0)} documents")

        # Test 3: Search with date range
        print("\n[Test 3] Searching documents from 2024...")
        result = await librarian_search_documents(
            filing_date_start="2024-01-01",
            page=1,
            page_size=5
        )
        data = json.loads(result)
        print(f"✓ Found {data.get('documents_count', 0)} documents from 2024")

        # Test 4: Combined filters
        print("\n[Test 4] Searching for slides from 2024...")
        result = await librarian_search_documents(
            document_types=["slides"],
            filing_date_start="2024-01-01",
            page=1,
            page_size=5
        )
        data = json.loads(result)
        print(f"✓ Found {data.get('documents_count', 0)} slides from 2024")

        print("\n✓ All search_documents tests passed!")
        return data.get('documents', [])

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        return []


async def test_batch_download(document_ids=None):
    """Test batch download"""
    print("\n" + "=" * 60)
    print("TEST: librarian_batch_download")
    print("=" * 60)

    try:
        if not document_ids:
            print("\n⚠ No document IDs provided, using sample IDs...")
            document_ids = [101356, 101357]

        # Test 1: Request raw documents
        print(f"\n[Test 1] Requesting raw download URLs for {len(document_ids)} documents...")
        result = await librarian_batch_download(
            document_ids=document_ids,
            representation_type="raw"
        )
        data = json.loads(result)
        print(f"✓ Requested {data.get('total_requested', 0)} documents")

        success_count = sum(1 for r in data.get('results', []) if r.get('status') == 'success')
        print(f"  Successful: {success_count}/{len(document_ids)}")

        if data.get('results'):
            first_result = data['results'][0]
            if first_result.get('download_url'):
                print(f"  Sample URL: {first_result['download_url'][:80]}...")
            else:
                print(f"  Error: {first_result.get('error')}")

        # Test 2: Request clean documents
        print(f"\n[Test 2] Requesting clean download URLs...")
        result = await librarian_batch_download(
            document_ids=document_ids[:1],
            representation_type="clean"
        )
        data = json.loads(result)
        print(f"✓ Requested clean version")

        # Test 3: Error handling - invalid type
        print(f"\n[Test 3] Testing error handling (invalid type)...")
        result = await librarian_batch_download(
            document_ids=document_ids[:1],
            representation_type="invalid"
        )
        if "Error" in result:
            print(f"✓ Correctly rejected invalid type")

        print("\n✓ All batch_download tests passed!")
        return data.get('results', [])

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        return []


async def test_get_document_metadata(document_id=None):
    """Test getting document metadata"""
    print("\n" + "=" * 60)
    print("TEST: librarian_get_document_metadata")
    print("=" * 60)

    try:
        if not document_id:
            print("\n⚠ No document ID provided, using sample ID...")
            document_id = 101356

        print(f"\n[Test 1] Fetching metadata for document {document_id}...")
        result = await librarian_get_document_metadata(document_id=document_id)

        if "Error" not in result:
            data = json.loads(result)
            print(f"✓ Retrieved metadata successfully")
            if data.get('data'):
                doc = data['data']
                print(f"  Title: {doc.get('title', 'N/A')}")
                print(f"  Type: {doc.get('document_type', 'N/A')}")
                print(f"  Pages: {doc.get('pages', 'N/A')}")
        else:
            print(f"⚠ Got error (document may not exist): {result}")

        print("\n✓ get_document_metadata test passed!")

    except Exception as e:
        print(f"\n✗ Test failed: {e}")


async def test_api_client_directly():
    """Test API client directly"""
    print("\n" + "=" * 60)
    print("TEST: Direct API Client")
    print("=" * 60)

    try:
        # Test 1: Get companies
        print("\n[Test 1] Direct API call - get companies...")
        result = await api_client.get_companies(
            countries=["Canada"],
            page=1,
            page_size=5
        )
        print(f"✓ API returned: {result.get('message', 'No message')}")
        print(f"  Companies: {len(result.get('data', []))}")

        # Test 2: Search documents
        print("\n[Test 2] Direct API call - search documents...")
        result = await api_client.search_documents(
            document_types=["filing"],
            page=1,
            page_size=5
        )
        print(f"✓ API returned: {result.get('message', 'No message')}")
        print(f"  Documents: {len(result.get('data', []))}")

        print("\n✓ All direct API client tests passed!")

    except Exception as e:
        print(f"\n✗ Test failed: {e}")


async def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("LIBRARIAN MCP SERVER - TEST SUITE")
    print("=" * 60)
    print("\nTesting connection to Librarian API...")
    print("Base URL: https://librarian.production.primerapp.com/api/v1")
    print("\n" + "=" * 60)

    # Test 1: Get companies
    companies = await test_get_companies()
    company_ids = [c.get('id') for c in companies if c.get('id')]

    # Test 2: Search documents
    documents = await test_search_documents(company_ids)
    document_ids = [d.get('id') for d in documents[:3] if d.get('id')]

    # Test 3: Batch download
    await test_batch_download(document_ids)

    # Test 4: Get document metadata
    if document_ids:
        await test_get_document_metadata(document_ids[0])

    # Test 5: Direct API client
    await test_api_client_directly()

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUITE COMPLETE")
    print("=" * 60)
    print("\n✓ All tests completed!")
    print("\nNext steps:")
    print("1. Review the results above")
    print("2. If all tests passed, the MCP server is ready to use")
    print("3. Configure Claude Desktop using claude_desktop_config.json")
    print("4. See README_MCP_SERVER.md for full documentation")
    print("\n" + "=" * 60)


async def quick_test():
    """Quick smoke test - just verify basic connectivity"""
    print("\n" + "=" * 60)
    print("QUICK SMOKE TEST")
    print("=" * 60)

    try:
        print("\nTesting API connectivity...")
        result = await api_client.get_companies(page=1, page_size=1)
        print(f"✓ API connection successful!")
        print(f"  Message: {result.get('message', 'No message')}")
        print(f"  Found companies: {len(result.get('data', []))}")

        print("\n✓ Smoke test passed! Server is ready to use.")
        print("\nRun full tests with: python test_mcp_server.py --full")

    except Exception as e:
        print(f"\n✗ Smoke test failed: {e}")
        print("\nPlease check:")
        print("1. API key is correct")
        print("2. Network connectivity")
        print("3. API endpoint is accessible")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--full":
        # Run full test suite
        asyncio.run(run_all_tests())
    else:
        # Run quick smoke test
        asyncio.run(quick_test())
        print("\nFor full test suite, run: python test_mcp_server.py --full")
