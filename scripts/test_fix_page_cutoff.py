#!/usr/bin/env python3
"""
Test script for fix_page_cutoff.py
"""

import asyncio
import sys
from pathlib import Path

# Add the script directory to path
sys.path.insert(0, str(Path(__file__).parent))

from fix_page_cutoff import PageCutoffFixer


async def test_find_missing_pages():
    """Test the find_missing_pages function."""
    fixer = PageCutoffFixer()
    
    # Test case 1: Missing page 30
    markdown_content = """
<!-- PAGE 29 -->

Some content

<!-- PAGE 31 -->

More content
"""
    
    missing_pages = fixer.find_missing_pages(markdown_content, 31)
    print(f"Test 1 - Missing pages for 31 total pages: {missing_pages}")
    assert 30 in missing_pages, "Page 30 should be missing"
    
    # Test case 2: Missing page 60 and last page
    markdown_content = """
<!-- PAGE 59 -->

Some content

<!-- PAGE 61 -->

More content
"""
    
    missing_pages = fixer.find_missing_pages(markdown_content, 62)
    print(f"Test 2 - Missing pages for 62 total pages: {missing_pages}")
    assert 60 in missing_pages, "Page 60 should be missing"
    assert 62 in missing_pages, "Page 62 should be missing"
    
    # Test case 3: No missing pages
    markdown_content = """
<!-- PAGE 30 -->

Some content

<!-- PAGE 60 -->

More content

<!-- PAGE 90 -->

More content
"""
    
    missing_pages = fixer.find_missing_pages(markdown_content, 90)
    print(f"Test 3 - Missing pages for 90 total pages: {missing_pages}")
    assert len(missing_pages) == 0, "No pages should be missing"
    
    print("✅ All tests passed!")


if __name__ == "__main__":
    asyncio.run(test_find_missing_pages())
