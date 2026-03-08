"""Load all FinQA dev PDF documents into an Elasticsearch index.

Usage:
    uv run load-finqa-to-es
    uv run load-finqa-to-es --index testing --batch-size 200
    uv run load-finqa-to-es --pdf-dir FinQA/test/pdf --no-tables
"""

import argparse
import os
from typing import List

from langchain_core.documents import Document

from config import load_environment_variables
from elastic.elasticsearch_utils import (
    build_elasticsearch_client,
    persist_documents_to_elasticsearch,
)
from index_store import load_documents_with_tables, resolve_pdf_files

DEFAULT_FINQA_DEV_DIR = "FinQA/test/pdf"
DEFAULT_INDEX = "testing"
DEFAULT_BATCH_SIZE = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load FinQA dev PDF documents into Elasticsearch."
    )
    parser.add_argument(
        "--pdf-dir",
        default=DEFAULT_FINQA_DEV_DIR,
        help=f"Root directory containing FinQA PDF files (default: {DEFAULT_FINQA_DEV_DIR}).",
    )
    parser.add_argument(
        "--index",
        "-i",
        default=os.getenv("ELASTICSEARCH_INDEX", DEFAULT_INDEX),
        help=f"Elasticsearch index name (default: {DEFAULT_INDEX}).",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Number of documents to index per bulk request (default: {DEFAULT_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--no-tables",
        action="store_true",
        help="Disable pdfplumber table extraction (index text only).",
    )
    return parser.parse_args()


def index_in_batches(
    client,
    documents: List[Document],
    index_name: str,
    batch_size: int,
) -> int:
    total = len(documents)
    indexed = 0
    for start in range(0, total, batch_size):
        batch = documents[start : start + batch_size]
        persist_documents_to_elasticsearch(client, batch, index_name=index_name)
        indexed += len(batch)
        print(f"  Indexed {indexed}/{total} documents into '{index_name}'...")
    return indexed


def main():
    load_environment_variables()
    args = parse_args()

    pdf_dir = args.pdf_dir
    if not os.path.isabs(pdf_dir):
        pdf_dir = os.path.join(os.getcwd(), pdf_dir)

    print(f"Scanning for PDFs under: {pdf_dir}")
    try:
        pdf_files = resolve_pdf_files(pdf_dir)
    except ValueError as error:
        print(f"Error: {error}")
        raise SystemExit(1)

    use_tables = not args.no_tables
    print(
        f"Loading {len(pdf_files)} PDF file(s) "
        f"({'with' if use_tables else 'without'} table extraction)..."
    )
    documents, total_tables = load_documents_with_tables(pdf_files, use_pdfplumber=use_tables)
    print(
        f"\nLoaded {len(documents)} document segments "
        f"({total_tables} table segments) from {len(pdf_files)} PDF(s)."
    )

    client = None
    try:
        client = build_elasticsearch_client()
        print(f"\nIndexing into Elasticsearch index '{args.index}' "
              f"in batches of {args.batch_size}...")
        total_indexed = index_in_batches(client, documents, args.index, args.batch_size)
        print(f"\nDone. {total_indexed} documents indexed into '{args.index}'.")
    except Exception as error:
        print(f"Failed to index documents: {error}")
        raise SystemExit(1)
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    main()
