import csv
import io
import os
from typing import List, Tuple

import pdfplumber  # type: ignore[import]
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

from chunking_strategies import (
    FixedSizeChunkingStrategy,
    RecursiveCharacterChunkingStrategy,
    SlidingWindowChunkingStrategy,
)
from config import load_environment_variables
from elastic.elasticsearch_utils import (
    build_elasticsearch_client,
    persist_documents_to_elasticsearch,
)
from vectore_store.vector_store import create_vector_store


def main():
    load_environment_variables()
    print("Creating vector stores...")

    document_path = os.getenv("DOCUMENT_PATH")
    if not document_path:
        print(
            "Error: DOCUMENT_PATH environment variable not set. Please set it in your .env file."
        )
        exit(1)

    try:
        pdf_files = resolve_pdf_files(document_path)
    except ValueError as error:
        print(error)
        exit(1)

    use_pdfplumber = os.getenv("USE_PDFPLUMBER", "true").lower() not in ("false", "0", "no")
    documents, total_tables = load_documents_with_tables(pdf_files, use_pdfplumber=use_pdfplumber)

    print(
        f"Total loaded {len(documents)} documents "
        f"(including {total_tables} extracted tables) from {len(pdf_files)} PDF(s)."
    )

    es_client = None
    try:
        es_client = build_elasticsearch_client()
        persist_documents_to_elasticsearch(es_client, documents)
        print("Documents persisted to Elasticsearch.")
    except Exception as elasticsearch_error:
        print(
            f"Warning: Failed to persist documents to Elasticsearch: {elasticsearch_error}"
        )
    finally:
        if es_client is not None:
            es_client.close()

    chunking_strategies = {
        "recursive": RecursiveCharacterChunkingStrategy(),
        "fixed_size": FixedSizeChunkingStrategy(),
        "sliding_window": SlidingWindowChunkingStrategy(),
    }

    table_suffix = "" if use_pdfplumber else "_no_tables"

    # Create or load vector stores for each strategy
    vector_stores = {}
    for name, strategy in chunking_strategies.items():
        index_name = f"faiss_index_{name}{table_suffix}"
        print(f"Creating vector store for {name} strategy (index: {index_name})...")
        vector_stores[name] = create_vector_store(documents, strategy, index_name=index_name)


def resolve_pdf_files(path: str) -> List[str]:
    if os.path.isdir(path):
        pdf_files = []
        for root, _, files in os.walk(path):
            for filename in files:
                if filename.lower().endswith(".pdf"):
                    pdf_files.append(os.path.join(root, filename))
        if not pdf_files:
            raise ValueError(f"No PDF files found within directory: {path}")
        print(f"Found {len(pdf_files)} PDF files under {path}")
        return sorted(pdf_files)
    if os.path.isfile(path) and path.lower().endswith(".pdf"):
        return [path]
    raise ValueError(f"Error: DOCUMENT_PATH '{path}' is not a valid PDF file or directory.")


def load_documents_with_tables(
    pdf_files: List[str], *, use_pdfplumber: bool = True
) -> Tuple[List[Document], int]:
    combined_documents: List[Document] = []
    total_tables = 0
    for filepath in pdf_files:
        print(f"\nProcessing {filepath} ...")
        text_docs = load_pdf_text_documents(filepath)
        combined_documents.extend(text_docs)
        if use_pdfplumber:
            table_docs = extract_table_documents(filepath)
            combined_documents.extend(table_docs)
            total_tables += len(table_docs)
            print(
                f"  - Added {len(text_docs)} text segments and {len(table_docs)} table segments."
            )
        else:
            print(f"  - Added {len(text_docs)} text segments (table extraction disabled).")
    return combined_documents, total_tables


def load_pdf_text_documents(filepath: str) -> List[Document]:
    loader = PyPDFLoader(filepath)
    return loader.load()


def extract_table_documents(filepath: str) -> List[Document]:
    table_docs: List[Document] = []
    try:
        with pdfplumber.open(filepath) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                tables = page.extract_tables()
                for table_index, rows in enumerate(tables):
                    if not rows:
                        continue
                    csv_content = rows_to_csv(rows)
                    table_docs.append(
                        Document(
                            page_content=csv_content,
                            metadata={
                                "source": filepath,
                                "page": page_number,
                                "table_index": table_index,
                                "table_rows": len(rows),
                                "table_cols": len(rows[0]) if rows[0] else 0,
                                "content_type": "table",
                                "table_extractor": "pdfplumber",
                            },
                        )
                    )
    except Exception as error:  # pragma: no cover - pdf-specific runtime issues
        print(f"Warning: pdfplumber failed to read tables from {filepath}: {error}")
        return []

    if table_docs:
        print(f"  - Extracted {len(table_docs)} table(s) with pdfplumber.")
    else:
        print("Warning: No tables detected by pdfplumber.")
    return table_docs


def rows_to_csv(rows: List) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    for row in rows:
        writer.writerow([(cell or "").strip() for cell in row])
    return buffer.getvalue()


if __name__ == "__main__":
    main()
