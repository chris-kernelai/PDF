#!/usr/bin/env python3
"""
Query Supabase database directly (READ-ONLY)

Fetches document IDs matching our criteria without using the API.
"""

import os
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()


def get_db_connection():
    """Create read-only database connection."""
    return psycopg2.connect(
        host=os.getenv("K_LIB_DB_HOST"),
        port=os.getenv("K_LIB_DB_PORT"),
        user=os.getenv("K_LIB_DB_USER"),
        password=os.getenv("K_LIB_DB_PASSWORD"),
        database=os.getenv("K_LIB_DB_NAME"),
        options="-c default_transaction_read_only=on",  # READ-ONLY mode
    )


def get_non_us_companies(conn):
    """Get all non-US company IDs."""
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            SELECT id, ticker, name, country
            FROM librarian.company
            WHERE country != 'United States'
            ORDER BY id
        """)
        companies = cur.fetchall()
        print(f"Found {len(companies)} non-US companies")
        return companies


def get_document_counts_by_id_range(conn, company_ids, max_doc_id=None):
    """Get document counts grouped by ID ranges."""
    # Convert to list for PostgreSQL array parameter
    company_ids_list = list(company_ids)

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        # Base query
        base_where = """
            WHERE company_id = ANY(%s::int[])
            AND document_type IN ('FILING', 'SLIDE')
        """

        # Query 1: All documents
        cur.execute(f"""
            SELECT
                COUNT(*) as total_count,
                COUNT(*) FILTER (WHERE document_type = 'FILING') as filing_count,
                COUNT(*) FILTER (WHERE document_type = 'SLIDE') as slides_count,
                MIN(id) as min_id,
                MAX(id) as max_id
            FROM librarian.kdocuments
            {base_where}
        """, (company_ids_list,))

        all_stats = cur.fetchone()

        print("\n" + "="*80)
        print("NON-US DOCUMENTS (filings + slides)")
        print("="*80)
        print(f"Total documents: {all_stats['total_count']:,}")
        print(f"  Filings: {all_stats['filing_count']:,}")
        print(f"  Slides: {all_stats['slides_count']:,}")
        print(f"Document ID range: {all_stats['min_id']:,} to {all_stats['max_id']:,}")

        if max_doc_id:
            print(f"\n" + "="*80)
            print(f"FILTERING BY max_doc_id <= {max_doc_id:,}")
            print("="*80)

            # Query 2: Documents with ID <= max_doc_id
            cur.execute(f"""
                SELECT
                    COUNT(*) as total_count,
                    COUNT(*) FILTER (WHERE document_type = 'FILING') as filing_count,
                    COUNT(*) FILTER (WHERE document_type = 'SLIDE') as slides_count,
                    MIN(id) as min_id,
                    MAX(id) as max_id
                FROM librarian.kdocuments
                {base_where}
                AND id <= %s
            """, (company_ids_list, max_doc_id))

            filtered_stats = cur.fetchone()

            print(f"Documents with ID <= {max_doc_id:,}: {filtered_stats['total_count']:,}")
            print(f"  Filings: {filtered_stats['filing_count']:,}")
            print(f"  Slides: {filtered_stats['slides_count']:,}")
            print(f"  ID range: {filtered_stats['min_id']:,} to {filtered_stats['max_id']:,}")

            filtered_out = all_stats['total_count'] - filtered_stats['total_count']
            print(f"\nFiltered out (ID > {max_doc_id:,}): {filtered_out:,}")

            # Query 3: Get actual document IDs for documents with ID <= max_doc_id
            print(f"\nFetching all document IDs with ID <= {max_doc_id:,}...")
            cur.execute(f"""
                SELECT id, document_type, company_id, published_at
                FROM librarian.kdocuments
                {base_where}
                AND id <= %s
                ORDER BY id
            """, (company_ids_list, max_doc_id))

            filtered_docs = cur.fetchall()

            print(f"Retrieved {len(filtered_docs):,} document records")

            # Show sample
            print(f"\nFirst 10 document IDs:")
            for doc in filtered_docs[:10]:
                print(f"  ID {doc['id']:6d} | {doc['document_type']:7s} | Company {doc['company_id']:5d} | {doc['published_at']}")

            print(f"\nLast 10 document IDs:")
            for doc in filtered_docs[-10:]:
                print(f"  ID {doc['id']:6d} | {doc['document_type']:7s} | Company {doc['company_id']:5d} | {doc['published_at']}")

            return filtered_docs

        return None


def get_all_document_stats(conn):
    """Get statistics for ALL documents (US + non-US)."""
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            SELECT
                COUNT(*) as total_count,
                COUNT(*) FILTER (WHERE document_type = 'FILING') as filing_count,
                COUNT(*) FILTER (WHERE document_type = 'SLIDE') as slides_count,
                COUNT(*) FILTER (WHERE document_type NOT IN ('FILING', 'SLIDE')) as other_count,
                MIN(id) as min_id,
                MAX(id) as max_id
            FROM librarian.kdocuments
        """)

        stats = cur.fetchone()

        print("\n" + "="*80)
        print("ALL DOCUMENTS IN DATABASE (US + non-US)")
        print("="*80)
        print(f"Total documents: {stats['total_count']:,}")
        print(f"  Filings: {stats['filing_count']:,}")
        print(f"  Slides: {stats['slides_count']:,}")
        print(f"  Other: {stats['other_count']:,}")
        print(f"Document ID range: {stats['min_id']:,} to {stats['max_id']:,}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Query Supabase database directly (READ-ONLY)")
    parser.add_argument(
        "--max-doc-id",
        type=int,
        help="Filter documents by maximum ID"
    )
    parser.add_argument(
        "--all-stats",
        action="store_true",
        help="Show statistics for ALL documents (US + non-US)"
    )

    args = parser.parse_args()

    print("Connecting to Supabase database (READ-ONLY mode)...")

    try:
        conn = get_db_connection()
        print("✅ Connected successfully\n")

        # Get all document stats first
        if args.all_stats:
            get_all_document_stats(conn)

        # Get non-US companies
        companies = get_non_us_companies(conn)
        company_ids = [c['id'] for c in companies]

        # Get document counts and IDs
        docs = get_document_counts_by_id_range(conn, company_ids, args.max_doc_id)

        if docs:
            # Save to file
            output_file = f"doc_ids_max_{args.max_doc_id}.txt"
            with open(output_file, 'w') as f:
                for doc in docs:
                    f.write(f"{doc['id']}\n")
            print(f"\n✅ Saved {len(docs):,} document IDs to {output_file}")

        conn.close()
        print("\n✅ Connection closed")

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
